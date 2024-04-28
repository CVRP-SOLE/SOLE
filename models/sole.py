from typing import OrderedDict
import torch
import hydra
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
import numpy as np
from datasets.class_labels import SCANNET200_CLASS_LABELS, SCANNET_CLASS_LABELS
import open_clip
import math

class SOLE(nn.Module):
    def __init__(
        self,
        config,
        backbone_weight,
        hidden_dim,
        clip_dim,
        class_embed,
        clip_model,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.clip_dim = clip_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type

        self.backbone = hydra.utils.instantiate(config.backbone)
        # initialize backbone with pretrained weight
        if self.training:
            checkpoint = torch.load(backbone_weight)
            new_checkpoint = OrderedDict()
            new_checkpoint["state_dict"] = OrderedDict()
            for k in checkpoint["state_dict"].keys():
                if "backbone" in k:
                    new_key = k.replace("model.backbone.", "")
                    new_checkpoint["state_dict"][new_key] = checkpoint["state_dict"][k]
            self.backbone.load_state_dict(new_checkpoint["state_dict"])
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        sizes = self.backbone.PLANES[-5:]
        self.num_levels = len(self.hlevels)
        
        self.semantic_proj = nn.Linear(self.clip_dim, self.mask_dim)
        self.clip_proj = nn.Linear(self.backbone.PLANES[7]+self.clip_dim, self.mask_dim)
        self.mask_features_head = conv(
            self.backbone.PLANES[7]+self.clip_dim,
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(
                mask, p2s, dim=dim
            )[0]
        else:
            assert False, "Scatter function not known"

        assert (
            not use_np_features
        ) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.clip_dim)

        if class_embed == 'scannet':
            CLASS_LABELS = SCANNET_CLASS_LABELS
        elif class_embed == 'scannet200':
            CLASS_LABELS = SCANNET200_CLASS_LABELS

        for i, c in enumerate(CLASS_LABELS):
            CLASS_LABELS[i] = "a " + CLASS_LABELS[i] + " in a scene."

        with torch.no_grad():
            clip, _, _ = open_clip.create_model_and_transforms(
                clip_model, "openai"
            )
            clip.cuda()
            clip.eval()
            tokenizer = open_clip.get_tokenizer(clip_model)

            text = tokenizer(CLASS_LABELS)
            textfeat = clip.encode_text(text.cuda())
            textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
            self.text_embed = textfeat

        self.bg_feature = nn.Parameter(torch.Tensor(1, self.clip_dim))
        nn.init.kaiming_uniform_(self.bg_feature, a=math.sqrt(5))
        self.bg_feature.requires_grad = True

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )
        else:
            assert False, "pos enc type not known"

        self.pooling = MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3
        )

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.cross_semantic_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_cross_semantic_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()

            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_cross_semantic_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(sizes[hlevel]+self.clip_dim, self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.cross_semantic_attention.append(tmp_cross_semantic_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(
        self, x, point2segment=None, raw_coordinates=None, clip_feats=None, mva_feats=None, mca_feats=None, mea_feats=None, is_eval=False
    ):  
        with torch.no_grad():
            batch_coords, batch_feats = ME.utils.sparse_collate(x.decomposed_coordinates, clip_feats)
            openseg_sparse = me.SparseTensor(
                coordinates = batch_coords,
                features = batch_feats,
                device=x.device,
            )

            openseg_aux = [openseg_sparse]
            for _ in reversed(range(len(self.hlevels))):
                openseg_aux.append(self.pooling(openseg_aux[-1]))
            openseg_aux.reverse()

            pcd_features, aux_ = self.backbone(x)
            aux = []
            for a, o in zip(aux_, openseg_aux):
                o._manager = a._manager
                aux.append(
                    me.cat(a, o)
                )

            if self.training:
                max_size = 0
                for mca_feat in mca_feats:
                    max_size = mca_feat.shape[0] if mca_feat.shape[0] > max_size else max_size

                semantic_key_mask = []
                semantic_key_idx = []
                key = []
                for mca_feat in mca_feats:
                    # key_size = c.shape[0] + e["key"].shape[0]
                    key_size = mca_feat.shape[0]

                    idx = torch.zeros(
                        max_size,
                        dtype=torch.long,
                        device=mca_feat.device,
                    )

                    midx = torch.ones(
                        max_size,
                        dtype=torch.bool,
                        device=mca_feat.device,
                    )

                    idx[:key_size] = torch.arange(
                        key_size, device=mca_feat.device
                    )

                    midx[:key_size] = False

                    key.append(mca_feat)
                    semantic_key_mask.append(midx)
                    semantic_key_idx.append(idx)

                batched_text_key = torch.stack(
                    [
                        key[k][semantic_key_idx[k], :]
                        for k in range(len(mca_feats))
                    ]
                )
                batched_text_key_mask = torch.stack(semantic_key_mask)
                batched_text_key_mask = batched_text_key_mask[..., None].repeat(1, 1, self.num_queries)
            else:
                batched_text_key = self.text_embed.unsqueeze(0).repeat(len(x.decomposed_coordinates), 1, 1)

        batched_text_key = self.semantic_proj(batched_text_key)
        pcd_features_clip = aux[-1]
        batch_size = len(x.decomposed_coordinates)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features_clip)
        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [
                furthest_point_sample(
                    x.decomposed_coordinates[i][None, ...].float(),
                    self.num_queries,
                )
                .squeeze(0)
                .long()
                for i in range(len(x.decomposed_coordinates))
            ]

            sampled_coords = torch.stack(
                [
                    coordinates.decomposed_features[i][fps_idx[i].long(), :]
                    for i in range(len(fps_idx))
                ]
            )

            mins = torch.stack(
                [
                    coordinates.decomposed_features[i].min(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )
            maxs = torch.stack(
                [
                    coordinates.decomposed_features[i].max(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )

            with torch.no_grad():
                backbone_features = aux[-1].decomposed_features
                queries = torch.stack(
                    [
                        backbone_features[i][fps_idx[i].long(), :]
                        for i in range(len(fps_idx))
                    ]
                )

            queries = self.clip_proj(queries)
            query_pos = self.pos_enc(
                sampled_coords.float(), input_range=[mins, maxs]
            )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            query_pos = (
                torch.rand(
                    batch_size,
                    self.mask_dim,
                    self.num_queries,
                    device=x.device,
                )
                - 0.5
            )

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (
                    torch.rand(
                        batch_size,
                        2 * self.mask_dim,
                        self.num_queries,
                        device=x.device,
                    )
                    - 0.5
                )
            else:
                query_pos_feat = torch.randn(
                    batch_size,
                    2 * self.mask_dim,
                    self.num_queries,
                    device=x.device,
                )

            queries = query_pos_feat[:, : self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim :, :].permute(
                (2, 0, 1)
            )
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(
                1, batch_size, 1
            )

        predictions_class = []
        predictions_mva = []
        predictions_mca = []
        predictions_mea = []
        predictions_mask = []
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                class_logit, mva_logit, mca_logit, mea_logit, outputs_mask, attn_mask = self.mask_module(
                    queries,
                    mask_features,
                    None,
                    len(aux) - hlevel - 1,
                    ret_attn_mask=True,
                    point2segment=None,
                    coords=coords,
                    mva_feats=mva_feats,
                    mca_feats=mca_feats,
                    mea_feats=mea_feats
                )

                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                curr_sample_size = max(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(
                        curr_sample_size, self.sample_sizes[hlevel]
                    )

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(
                            pcd_size, device=queries.device
                        )

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(
                            decomposed_aux[k].shape[0], device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [
                        decomposed_aux[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn = torch.stack(
                    [
                        decomposed_attn[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](
                    batched_aux.permute((1, 0, 2))
                )
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                if self.training:
                    output = self.cross_semantic_attention[decoder_counter][i](
                        output,
                        batched_text_key.permute((1, 0, 2)),
                        memory_mask = batched_text_key_mask.repeat_interleave(
                            self.num_heads, dim=0
                        ).permute((0, 2, 1)),
                        query_pos=query_pos,
                    )
                else:
                    output = self.cross_semantic_attention[decoder_counter][i](
                        output,
                        batched_text_key.permute((1, 0, 2)),
                        query_pos=query_pos,
                    )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                predictions_class.append(class_logit)
                predictions_mva.append(mva_logit)
                predictions_mca.append(mca_logit)
                predictions_mea.append(mea_logit)
                predictions_mask.append(outputs_mask)

        class_logit, mva_logit, mca_logit, mea_logit, outputs_mask = self.mask_module(
            queries,
            mask_features,
            None,
            0,
            ret_attn_mask=False,
            point2segment=None,
            coords=coords,
            mva_feats=mva_feats,
            mca_feats=mca_feats,
            mea_feats=mea_feats
        )
        
        predictions_class.append(class_logit)
        predictions_mva.append(mva_logit)
        predictions_mca.append(mca_logit)
        predictions_mea.append(mea_logit)
        predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_mva_logits": predictions_mva[-1],
            "pred_mca_logits": predictions_mca[-1],
            "pred_mea_logits": predictions_mea[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, 
                predictions_mva,
                predictions_mca, 
                predictions_mea,
                predictions_mask
            ),
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
        }

    def mask_module(
        self,
        query_feat,
        mask_features,
        mask_segments,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        coords=None,
        mva_feats=None,
        mca_feats=None,
        mea_feats=None
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        outputs_class = torch.nn.functional.normalize(outputs_class, dim=-1)

        mva_logit = []
        for o, mva_feat in zip(outputs_class, mva_feats):
            class_cost = o @ mva_feat.t() * 100
            bg_cost = o @ self.bg_feature.t() * 100
            mva_logit.append(torch.cat([class_cost, bg_cost], dim=-1))

        mca_logit = []
        for o, mca_feat in zip(outputs_class, mca_feats):
            class_cost = o @ mca_feat.t() * 100
            bg_cost = o @ self.bg_feature.t() * 100
            mca_logit.append(torch.cat([class_cost, bg_cost], dim=-1))

        mea_logit = []
        for o, mea_feat in zip(outputs_class, mea_feats):
            class_cost = o @ mea_feat.t() * 100
            bg_cost = o @ self.bg_feature.t() * 100
            mea_logit.append(torch.cat([class_cost, bg_cost], dim=-1))

        if not self.training:
            class_cost = outputs_class @ self.text_embed.t() * 100
            bg_cost = outputs_class @ self.bg_feature.t() * 100
            class_logit = torch.cat([class_cost, bg_cost], dim=-1)
        else:
            class_logit = None

        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(
                    mask_features.decomposed_features[i] @ mask_embed[i].T
                )

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            if point2segment is not None:
                return class_logit, mva_logit, mca_logit, mea_logit, output_segments, attn_mask
            else:
                return (
                    class_logit,
                    mva_logit,
                    mca_logit,
                    mea_logit,
                    outputs_mask.decomposed_features,
                    attn_mask
                )

        if point2segment is not None:
            return class_logit, mva_logit, mca_logit, mea_logit, output_segments
        else:
            return class_logit, mva_logit, mca_logit, mea_logit, outputs_mask.decomposed_features

    @torch.jit.unused
    def _set_aux_loss(
        self, 
        class_logit, 
        mva_class, 
        mca_class, 
        mea_class,
        outputs_seg_masks
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a, 
                "pred_mva_logits": b, 
                "pred_mca_logits": c, 
                "pred_mea_logits": d,
                "pred_masks": e
                }
            for a, b, c, d, e in zip(
                class_logit[:-1], 
                mva_class[:-1],
                mca_class[:-1],
                mea_class[:-1],
                outputs_seg_masks[:-1]
            )
        ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
