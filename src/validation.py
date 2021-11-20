import glob
from validation.densetnt_validation import DenseTNTValidation

if __name__ == "__main__":
    max_workers = 5
    model_path = "models/v4_3/model.30.bin"
    data_folder = "/home/anhtt163/dataset/OBP/data_test"
    # iter through all batch test in folder
    for batch in glob.glob(f"{data_folder}/*"):
        map_path = f"{batch}/static.csv"
        denseTNT_validation = DenseTNTValidation(
            map_path=map_path, model_path=model_path,
            max_workers=max_workers
        )

        dynamic_folder = f"{batch}/dynamic_by_ts"
        mfde, mr = denseTNT_validation.run(dynamic_folder)

        print(f"Score for {batch}")
        print("Mean FDE:", mfde)
        print("Miss rate:", mr)
