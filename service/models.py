import typing as tp
import dill
import pickle

from pydantic import BaseModel

from service.settings import get_config


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class Popular:
    """Class for predict Popular model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/popular_answer.dill", "rb") as f:
            self.answer = dill.load(f)

    def predict(self) -> list:
        return self.answer


class userKNN:
    """Class for predict kNN model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/userknn.dill", "rb") as f:
            self.user_knn = dill.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.user_knn.users_mapping:
            reco = self.user_knn.eval(user_id, N_recs=self.N_recs).item_id.to_list()
            if len(reco) < self.N_recs:
                reco_popular = self.popular_model.predict()
                reco += [item for item in reco_popular if item not in reco][: self.N_recs - len(reco)]
        else:
            reco = self.popular_model.predict()
        return reco


class userKNNOffline:
    """Class for offline kNN model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/userknn_offline.dill", "rb") as f:
            self.user_knn_pred_result = dill.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.user_knn_pred_result:
            reco = self.user_knn_pred_result[user_id][:self.N_recs]
            if len(reco) < self.N_recs:
                reco_popular = self.popular_model.predict()
                reco += [item for item in reco_popular if item not in reco][: self.N_recs - len(reco)]
        else:
            reco = self.popular_model.predict()
        return reco


class ALSOffline:
    """Class for offline ALS model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/als_predict_offline.dill", "rb") as f:
            self.als_pred_result = dill.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.als_pred_result:
            reco = self.als_pred_result[user_id][:self.N_recs]
        else:
            reco = self.popular_model.predict()
        return reco

class DSSMOffline:
    """Class for offline DSSM model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/dssm_predict_offline.pkl", "rb") as f:
            self.dssm_pred_result = pickle.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.dssm_pred_result:
            reco = self.dssm_pred_result[user_id][:self.N_recs]
        else:
            reco = self.popular_model.predict()
        return reco


class AutoencoderOffline:
    """Class for offline Autoencoder model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/autoencoder_offline.pkl", "rb") as f:
            self.autoencoder_pred_result = pickle.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.autoencoder_pred_result:
            reco = self.autoencoder_pred_result[user_id][:self.N_recs]
        else:
            reco = self.popular_model.predict()
        return reco


class RecboleOffline:
    """Class for offline Recbole model"""

    def __init__(self, N_recs: int = 10):
        self.N_recs = N_recs

        with open("./service/models_folder/recbole_offline.pkl", "rb") as f:
            self.recbole_pred_result = pickle.load(f)

        self.popular_model = Popular(self.N_recs)

    def predict(self, user_id: int) -> list:
        if user_id in self.recbole_pred_result:
            reco = self.recbole_pred_result[user_id][:self.N_recs]
        else:
            reco = self.popular_model.predict()
        return reco


app_config = get_config()

popular = Popular(N_recs=app_config.k_recs)
user_knn = userKNNOffline(N_recs=app_config.k_recs)
als = ALSOffline(N_recs=app_config.k_recs)
dssm = DSSMOffline(N_recs=app_config.k_recs)
autoencoder = AutoencoderOffline(N_recs=app_config.k_recs)
recbole = RecboleOffline(N_recs=app_config.k_recs)
recbole_onl = RecboleOnline(N_recs=app_config.k_recs)