import yaml
from data.data_ingestion import data_ingestion
from exploration.descriptive_stats import descriptive_stats
from preprocessing.preprocessing import preprocess_data
from models.model import build_model, evaluate_model




def run_pipeline(config_path="configs/config.yaml"):
    cfg = yaml.safe_load(open(config_path))


    # Step1: Ingestion
    df = data_ingestion(cfg['data_path'])


    # Step2: Exploration
    num_stats, cat_stats, info = descriptive_stats(df)


    # Step3: Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df,cfg['target_col'],
                                                    cfg['test_size'],
                                                    cfg['random_state'])


    # Step4: Model
    model = build_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)


    print("Model R2 Score:", score)
    return score