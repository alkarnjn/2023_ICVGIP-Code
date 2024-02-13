#%% INIT
from pymongo import MongoClient,TEXT

from lib.globals import *

db = MongoClient(MONGO_URI)
db = db[MONGO_DB]

#%% FVC2000
from re import compile as r
# coll = db["FVC2000"]
db_names = [
    "FVC2000",
    "FVC2002",
    # "FVC2004",
    # "FVC2006",
]
sub_dbs = [
    "Db1",
    "Db2",
    "Db3",
    "Db4",
]

#%% Testing index
def gen_update(docs,sub_db):
    for doc in docs:
        doc['path']=doc['path'].replace(sub_db,sub_db.upper()).replace("_a","_A").replace("_b","_B")
        yield doc
for db_name in db_names:
    for sub_db in sub_dbs:
        coll = db[db_name]
        # sub_db=sub_db.upper()
# db_name = db_names[0]
# sub_db = sub_dbs[0]
        new_coll = db[f"{db_name}_{sub_db.upper()}"]
        new_coll.drop()
# new_coll.create_index([("path",TEXT)])
        new_coll.create_index("path", unique=True)
        filter = {
            "path": {"$regex": f"{db_name}/Dbs/{sub_db}"},
        }
        new_filter = {
            "path": {"$regex": f"{db_name}/Dbs/{sub_db.upper()}"},
        }
        Found=coll.count_documents(filter)
        print(f"{Found=}")
        
             
        if Found>0 and new_coll.count_documents(new_filter) == 0:
            result=new_coll.insert_many(gen_update(coll.find(filter),sub_db))
            print(f"{len(result.inserted_ids)=}")
        if Found==0:
            Found=coll.count_documents(new_filter)
            print(f"{Found=}")
            if Found>0 and new_coll.count_documents(new_filter) == 0:
                result=new_coll.insert_many(coll.find(new_filter))
                print(f"{len(result.inserted_ids)=}")
             



# %% Update folder to uppercase
from pathlib import Path
for db_name in db_names:
    base=Path(f"/home/rs/19CS91R05/datasets/{db_name}/Dbs")
    dirs=list([d for d in base.iterdir() if d.is_dir()])
    print(dirs)

    for d in dirs:
        d.rename(d.parent/(d.name.upper()))
#%% update paths for FVC2000 and FVC2002 in mongodb

db_names = [
    "FVC2000",
    "FVC2002",
]
sub_dbs = [
    "Db1",
    "Db2",
    "Db3",
    "Db4",
]

for db_name in db_names:
    for sub_db in sub_dbs:
        coll = db[db_name]
        print(f"Working for {db_name} {sub_db}")
        # sub_db=sub_db.upper()
        filter = {
            "path": {"$regex": f"{db_name}/Dbs/{sub_db}"},
        }
        Found=coll.count_documents(filter)
        print(f"{Found=}")

        for doc in gen_update(coll.find(filter,{"path":1}),sub_db):
            # doc['path']=doc['path'].replace("Db","FVC2000/DB1").replace("_a","_A").replace("_b","_B")
            coll.update_one({"_id":doc["_id"]},{"$set":{"path":doc["path"]}})
# for doc in docs:
#     doc['path']=doc['path'].replace("FVC2000","FVC2000/DB1").replace("_a","_A").replace("_b","_B")
#     yield doc

# list(db.list_collections())