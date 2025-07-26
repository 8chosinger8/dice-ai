import pytest
from game_of_dice_super_ai_sss import SLevelAIPredictor

def test_train_models_runs():
    ai = SLevelAIPredictor()
    dummy_data = [[i, 0, '大' if i % 2 == 0 else '小', "", "", "", "", None] for i in range(20)]
    success = ai.train_models(dummy_data)
    assert isinstance(success, bool)

def test_build_and_train_lstm_runs():
    import numpy as np
    ai = SLevelAIPredictor()
    X = np.random.rand(20, 10)
    y = np.random.randint(0, 2, size=20)
    model = ai.build_and_train_lstm(X, y, epochs=1, batch_size=4)
    assert model is not None

def test_predict_with_confidence_runs():
    ai = SLevelAIPredictor()
    dummy_results = ["大", "小", "大", "大", "小"]
    pred, conf, expl = ai.predict_with_confidence(dummy_results)
    assert pred in ["大", "小"]
    assert isinstance(conf, (int, float))
    assert isinstance(expl, str)
