from dpt.models import DPTDepthModel

config = {
    "base_scale": 0.0000305,
    "base_shift": 0.1378,
    "batch_size": 1,
    "image_size": (384, 384),
    "early_stopping_patience": 10,
    "num_workers": 0,
    "model_path": f"./models/2022_07_05_10_05_31.pt",
}
import cv2
import torch
from torchvision.transforms import Compose
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import matplotlib.pyplot as plt


def read_image(path):
    img = cv2.imread(str(path))

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
def main(image_paths):
    num_images = len(image_paths)
    model = DPTDepthModel(
        path=config["model_path"],
        scale=config["base_scale"],
        shift=config["base_shift"],
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()

    print("start processing")
    for idx, img_name in enumerate(image_paths):
        print("  processing {} ({}/{})".format(img_name, idx + 1, num_images))
        # input

        img = read_image(img_name)

        # if args.kitti_crop is True:
        #     height, width, _ = img.shape
        #     top = height - 352
        #     left = (width - 1216) // 2
        #     img = img[top : top + 352, left : left + 1216, :]

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).unsqueeze(0)

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            print(prediction)
            plt.imshow(prediction)
            # plt.imshow(mask, alpha=0.7)
            plt.show()
        #     if model_type == "dpt_hybrid_kitti":
        #         prediction *= 256

        #     if model_type == "dpt_hybrid_nyu":
        #         prediction *= 1000.0

        # filename = os.path.join(
        #     output_path, os.path.splitext(os.path.basename(img_name))[0]
        # )
        # util.io.write_depth(
        #     filename, prediction, bits=2, absolute_depth=args.absolute_depth
        # )

    print("finished")


from pathlib import Path

samples_base = "/Users/taichi.muraki/workspace/machine-learning/mur6-lightning-flash-test/data/samples"
samples_base = Path(samples_base)

main(list(samples_base.glob("*.jpeg")))
