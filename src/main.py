from preprocess import run_preprocess
from train import run_training
from register import run_registry
from export_model import export_model


def main():

    print("STEP 1: Preprocessing")
    run_preprocess()

    print("STEP 2: Training")
    run_training()

    print("STEP 3: Registering model")
    run_registry()

    print("STEP 4: Exporting production model to app folder")
    export_model()

    print("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()