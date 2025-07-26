# test_dice_ai.py
import pytest
from game_of_dice_super_ai_sss import SLevelAIPredictor  # 替換為您的檔案名

def test_train_models():
    ai = SLevelAIPredictor()
    historical_data = [("1", 8, "小")] * 25  # 模擬25局數據
    assert ai.train_models(historical_data) is True  # 測試訓練成功

def test_predict_with_confidence():
    ai = SLevelAIPredictor()
    historical_results = ["小"] * 20
    predicted, conf, expl = ai.predict_with_confidence(historical_results)
    assert predicted in ["大", "小"]
    assert 0 <= conf <= 100
