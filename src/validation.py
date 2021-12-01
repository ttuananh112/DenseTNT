import json
import carla

from validation.densetnt_validation import DenseTNTValidation
from validation.purepursuit_validation import PurePursuitValidation


def get_config(config_path: str):
    with open(config_path, "r") as f:
        data_config = json.load(f)
    return data_config


def get_topology(host, port, town):
    client = carla.Client(host=host, port=port)
    world = client.load_world(town)
    map = world.get_map()
    topology = map.get_topology()
    return topology


if __name__ == "__main__":
    max_workers = 5
    model_path = "models/v9_50m/model.30.bin"
    data_folder = "/home/anhtt163/dataset/OBP/data_test"
    batch = f"{data_folder}/all_batches"

    result = dict()
    result[batch] = dict()
    map_path = f"{batch}/static.csv"
    dynamic_folder = f"{batch}/dynamic_by_ts"

    # configuration
    config = get_config(f"{batch}/data_config.txt")
    host = config["connection"]["host"]
    port = config["connection"]["port"]
    town = config["map"]["town"]

    # calculate score for DenseTNT
    mfde, mr = DenseTNTValidation(
        map_path=map_path, model_path=model_path,
        max_workers=max_workers
    ).run(dynamic_folder, debug=None)
    result[batch]["DenseTNT"] = {"mfde": mfde, "mr": mr}

    # # calculate score for Pure-Pursuit
    # mfde, mr = PurePursuitValidation(
    #     topology=get_topology(host, port, town)
    # ).run(dynamic_folder)
    # result[batch]["PurePursuit"] = {"mfde": mfde, "mr": mr}

    print(json.dumps(result, sort_keys=True, indent=4))
