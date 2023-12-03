import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from dateutil import parser
from pandarallel import pandarallel
import spacy 
import pgeocode
from sklearn.model_selection import train_test_split

pandarallel.initialize(progress_bar=True)
tqdm.pandas()

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer

nomi = pgeocode.Nominatim('us')

data_path = Path(r"D:\Productivity\Studying\PMLDL_A2\data\raw\ml-100k")
interim_path = Path(r"D:\Productivity\Studying\PMLDL_A2\data\interim")


def encode_date(date):
    date = parser.parse(date)
    return np.sin(date.day), np.cos(date.day), np.sin(date.month), np.cos(date.month), date.year

def split_title(title):
    *actural_title, release = title.split()
    release = release.removeprefix("(").removesuffix(")")
    release = int(release) if release.isdigit() else np.nan
    actural_title = " ".join(actural_title)
    return actural_title, release

def embed(title):
    return nlp(tokenizer(title)).vector


def process_datasets(df):
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    
    df_user = pd.read_csv(data_path / "u.user", sep= "|", encoding='latin-1', header=None, 
                          names=["id", "age", "gender", "occupation", "zip_code"])
    item_df = pd.read_csv(data_path / "u.item", sep= "|", encoding='latin-1', header=None)
    df_occupation = pd.read_csv(data_path / "u.occupation", sep= "|", encoding='latin-1', header=None)

    df_user = df_user[df_user.id.isin(df.user_id)]


    df["timestamp"] = df["timestamp"].apply(lambda x: pd.Timestamp(x, unit="s"))


    item_df = item_df.drop(columns=[0, 3, 4])
    item_df = item_df.dropna()

    item_df[["title", "year1"]] = item_df[1].progress_apply(split_title).progress_apply(pd.Series)
    item_df.dropna(inplace=True)
    item_df.drop(columns=[1], inplace=True)

    embeddings = item_df["title"].progress_apply(embed).parallel_apply(pd.Series)
    item_df.drop(columns=["title"], inplace=True)
    item_df["year1"] = (item_df["year1"]-item_df["year1"].min())/(item_df["year1"].max()-item_df["year1"].min())
    item_df = pd.concat([item_df, embeddings], axis=1, ignore_index=True)

    date_features = item_df[0].progress_apply(encode_date).parallel_apply(pd.Series)
    date_features[4] = (date_features[4]-date_features[4].min())/(date_features[4].max()-date_features[4].min())

    item_df.drop(columns=[0], inplace=True)
    item_df = pd.concat([item_df, date_features], axis=1, ignore_index=True)

    occupation_dtype = pd.CategoricalDtype(categories=df_occupation[0].to_list())
    df_user["occupation"] = pd.Series(df_user["occupation"], dtype=occupation_dtype)
    df_occupation = pd.get_dummies(df_user["occupation"], dtype=float)

    df_geo = df_user.zip_code.parallel_apply(nomi.query_postal_code)
    coordinates = df_geo[["latitude", "longitude"]]

    df_user.age = (df_user.age-df_user.age.min())/(df_user.age.max()-df_user.age.min())
    df_user.gender = df_user.gender.map({"M": 0, "F": 1})

    df_user[["latitude", "longitude"]] = coordinates
    df_user.drop(columns=["zip_code", "occupation"], inplace = True)
    df_user = pd.concat([df_occupation, df_user], axis=1)
    df_user.rename(columns={"id": "user_id"}, inplace=True)
    df_user = df_user.set_index("user_id", drop=True)

    item_df["item_id"] = item_df.index + 1
    item_df.set_index("item_id", drop=True, inplace=True)

    df = df.join(item_df, on="item_id").join(df_user, on="user_id")
    df.drop(columns=["user_id", "item_id", "timestamp"], inplace=True)
    df.dropna(inplace=True)

    return df.drop(columns="rating"), df.rating / 5