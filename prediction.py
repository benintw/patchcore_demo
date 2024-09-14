import torch
import matplotlib.pyplot as plt
from PIL import Image
import config
from icecream import ic


def predict_on_test_set(
    testing_defect_folder_path, transform, backbone, memory_bank, best_threshold
):
    """Predict on the test set and display results

    :Example:
    >>>
    testing_defect_folder_path: mvtec/{SUBJECT}/test/{DEFECT_TYPE}

    """

    backbone.eval()

    # for path in testing_defect_folder_path.glob("*/*.png"):
    #     fault_type = path.parts[-2]

    #     if fault_type in ["metal_contamination"]:
    #         ...

    defect_type = testing_defect_folder_path.parts[-1]

    for path in testing_defect_folder_path.iterdir():

        if path.suffix.endswith(".png"):

            ic(path.name)

            test_image = transform(Image.open(path)).to(config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                features = backbone(test_image)

            distances = torch.cdist(features, memory_bank, p=2.0)

            dist_score, dist_score_idxs = torch.min(distances, dim=1)
            s_star = torch.max(dist_score)
            segm_map = dist_score.view(1, 1, 28, 28)

            segm_map = (
                torch.nn.functional.interpolate(
                    segm_map, size=(224, 224), mode="bilinear"
                )
                .cpu()
                .squeeze()
                .numpy()
            )

            y_score_image = s_star.cpu().numpy()

            y_pred_image = 1 * (y_score_image >= best_threshold)
            class_label = ["OK", "NOK"]

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
            plt.title(f"Fault type: {defect_type}")

            plt.subplot(1, 3, 2)
            heat_map = segm_map
            plt.imshow(
                heat_map, cmap="jet", vmin=best_threshold, vmax=best_threshold * 2
            )
            plt.title(
                f"Anomaly score: {y_score_image / best_threshold:0.4f} || {class_label[y_pred_image]}"
            )

            plt.subplot(1, 3, 3)
            plt.imshow((heat_map > best_threshold * 1.25), cmap="gray")
            plt.title("Segmentation map")

            plt.show()

    backbone.train()
