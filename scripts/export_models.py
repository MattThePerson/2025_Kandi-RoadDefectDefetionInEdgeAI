import shutil
from pathlib import Path
import os

import ultralytics

os.chdir('..')

""" arguments """
image_sizes = [640, 320]  # export image sizes
base_dir = 'runs_puhti'  # base folder for model directories

filters = [
    # 'long_runs',
    # '150ep',
    # 'normal',
]
run_paths = sorted(
    Path(base_dir).rglob('best.pt'),
    key=lambda obj: str(obj),
)
run_paths = [ p.parent.parent for p in run_paths ]
for f in filters:
    run_paths = [ p for p in run_paths if f in str(p) ]

new_paths = []

print("Exporting models: ", run_paths)
print(f"Exporting to image sizes: {image_sizes}")
print(f"{len(run_paths)} (models) x {len(image_sizes)} (image sizes) = {len(run_paths)*len(image_sizes)} exported models")

def export(imgsz):
# export models
    for i, model_dir in enumerate(run_paths):
        model_path = model_dir.joinpath("weights/best.pt")

        quant_dir = model_dir.with_name(f"{model_dir.name}.quant.{imgsz}")

        # check if the model is already exported
        if quant_dir.joinpath("best_saved_model").joinpath("best_full_integer_quant.tflite").exists():
            print(f"\nExported model {quant_dir.joinpath('best_full_integer_quant.tflite')} already exists.")
            continue

        # load model
        model = ultralytics.YOLO(model_path)

        # e = ultralytics.YOLO.export(model, format="edgetpu", imgsz=imgsz)
        # print(type(e))
        # break

        # export
        try:
            export_path = ultralytics.YOLO.export(model, format="edgetpu", imgsz=imgsz)
        except Exception as e:
            print(e)
            print(f"Export failed for model {model_dir.name} with imgsz {imgsz}:\n {e}")
            continue


        # Create new directory for quantized model if export was successful
        new_paths.append(quant_dir.name)  # keep a list of new directories

        quant_dir.mkdir(exist_ok=True)
        shutil.move(str(Path(export_path).parent), str(quant_dir))

        print(f"\nSuccessfully exported model: {model_dir.name} -> {quant_dir.name}\n")

        # rename best_saved_model -> weights

for imgsz in image_sizes:
    export(imgsz)

print(f"Exporting finished, created {len(new_paths)} models: \n", new_paths)