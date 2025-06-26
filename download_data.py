from huggingface_hub import snapshot_download
import os
import shutil

PROCESSED_DATA = "data"
BACKBONE_CHECKPOINT = "backbone_checkpoint"
CHECKPOINT = "checkpoint"
OPENVOCAB = "openvocab_supervision"

def download_from_huggingface():
    snapshot_download(
        repo_id="onandon/sole",
        repo_type="dataset",
        local_dir="cache",
        local_dir_use_symlinks=False
    )

def setting():
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("backbone_checkpoint", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("openvocab_supervision", exist_ok=True)

    shutil.move("cache/backbone_scannet.ckpt", BACKBONE_CHECKPOINT)
    shutil.move("cache/backbone_scannet200.ckpt", BACKBONE_CHECKPOINT)
    shutil.move("cache/scannet.ckpt", CHECKPOINT)
    shutil.move("cache/scannet200.ckpt", CHECKPOINT)
    shutil.move("cache/replica.ckpt", CHECKPOINT)
    shutil.move("cache/processed", PROCESSED_DATA)
    shutil.move("cache/openseg", OPENVOCAB)
    shutil.move("cache/scannet_mca", OPENVOCAB)
    shutil.move("cache/scannet_mea", OPENVOCAB)
    shutil.move("cache/scannet200_mca", OPENVOCAB)
    shutil.move("cache/scannet200_mea", OPENVOCAB)

    shutil.rmtree("cache")

if __name__ == "__main__":
    print("INFO : Downloading the preprocessed data from huggingface...")
    download_from_huggingface()
    print("INFO : Moving the donwloaded data...")
    setting()
    print("INFO : Now, you are ready to run SOLE.")
