import pandas as pd
import torch
import sqlite3
import re
import os
import os.path as osp

from nebula.common import get_device
from nebula.data.nvbench.process_dataset import ProcessData4Training
from nebula.data.nvbench.setup_data_bert import get_bert_tokenizer, setup_data
from nebula.model.nv_bert.component.bert_encoder import BertEncoder, EMBEDDING_SIZE
from nebula.model.nv_bert.component.decoder import Decoder
from nebula.model.nv_bert.component.seq2seq import Seq2Seq
from nebula.model.nv_bert.translate import get_token_types, fix_chart_template


# TODO: incomplete, finish translation with guidance
def translate(input_src, model, label_vocab, device):
    model.eval()

    tokenizer = get_bert_tokenizer()
    res = tokenizer(input_src, return_tensors="pt")
    src_tensor = res["input_ids"]
    src_mask = res["attention_mask"]

    with torch.no_grad():
        enc_src = \
            model.encoder(src_tensor, src_mask)

    trg_indexes = [label_vocab['<sos>']]
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)
    with torch.no_grad():
        output, attention = \
            model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # append output to trg_indexes and predict again.

    return None


class nvBert:
    def __init__(
            self,
            trained_model_path=None,
            temp_dataset_path=".",
            batch_size=128,
    ):
        self.temp_dataset_path = temp_dataset_path
        self.device = get_device()

        (
            self.train_dl,
            self.validation_dl,
            self.test_dl,
            self.train_dl_small,
            self.label_vocab
        ) = setup_data(batch_size=batch_size)

        OUTPUT_DIM = len(self.label_vocab.vocab)
        HID_DIM = EMBEDDING_SIZE  # it equals to embedding dimension
        DEC_LAYERS = 3
        DEC_HEADS = 8
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        MAX_LENGTH = 128

        enc = BertEncoder(dropout=ENC_DROPOUT)

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device,
                      MAX_LENGTH
                      )

        bert_tokenizer = get_bert_tokenizer()
        self.SRC_PAD_IDX = bert_tokenizer.pad_token_id
        self.TRG_PAD_IDX = self.label_vocab.get_stoi()["<pad>"]

        self.model = Seq2Seq(
            enc,
            dec,
            self.SRC_PAD_IDX,
            self.TRG_PAD_IDX,
            self.device
        ).to(self.device)

    def translate(
            self,
            input_src,
            token_types,
            db_id=None,
            table_name=None,
            db_tables_columns=None,
            db_tables_columns_types=None
    ):

        db_id = db_id or self.db_id
        table_name = table_name or self.table_id
        db_tables_columns = db_tables_columns or self.db_tables_columns
        db_tables_columns_types = db_tables_columns_types or self.db_tables_columns_types

        res = translate(input_src, self.model, self.label_vocab, self.device)

        return res

    def predict(
            self,
            nl_question,
            chart_template=None,
            show_progress=None,
            visualization_aware_translation=True
    ):

        input_src, token_types = self.process_input(nl_question, chart_template)

        res = self.translate(
            input_src=input_src,
            token_types=token_types
        )

        return res

    def process_input(self, nl_question, chart_template):

        query_template = fix_chart_template(chart_template)

        # get a list of mentioned values in the NL question
        col_names, value_names = self.data_processor.get_mentioned_values_in_NL_question(
            self.db_id, self.table_id, nl_question, db_table_col_val_map=self.db_table_col_val_map
        )
        col_names = ' '.join(str(e) for e in col_names)
        value_names = ' '.join(str(e) for e in value_names)

        input_src = (
            f"<N> {nl_question} </N> " \
            f"<C> {query_template} </C> " \
            f"<D> {self.table_id} <COL> {col_names} </COL> <VAL> {value_names} </VAL> </D>").lower()
        token_types = get_token_types(input_src)

        return input_src, token_types

    def specify_dataset(
            self,
            data_type,
            db_url = None,
            table_name = None,
            data = None,
            data_url = None
    ):
        '''
        this function creates a temporary save db for the input data
        the save db is a sqlite db in ./dataset/database
        the db name is temp_<table_name>, there is one table in it called <table_name>

        :param data_type: sqlite3, csv, json
        :param db_url: db path for sqlite3 database,
                       e.g., './dataset/database/flight/flight.sqlite'
        :param table_name: the table name in a sqlite3
        :param data: DataFrame for csv
        :param data_url: data path for csv or json
        :return: save the DataFrame in the self.data
        '''
        self.db_id = 'temp_' + table_name
        self.table_id = table_name

        # read in data as dataframe
        if data_type == 'csv':
            if data != None and data_url == None:
                self.data = data
            elif data == None and data_url != None:
                self.data = pd.read_csv(data_url)
            else:
                raise ValueError('Please only specify one of the data or data_url')
        elif data_type == 'json':
            if data == None and data_url != None:
                self.data = pd.read_json(data_url)
            else:
                raise ValueError(
                    'Read JSON from the json file, ' 
                    'please only specify the "data_type" or "data_url"'
                )
        elif data_type == 'sqlite3':
            # Create your connection.
            try:
                cnx = sqlite3.connect(db_url)
                self.data = pd.read_sql_query("SELECT * FROM " + table_name, cnx)
            except:
                raise ValueError(
                    f'Errors in read table from sqlite3 database. \n' 
                    f'db_url: {db_url}\n'
                    f' table_name : {table_name} '
                )
        else:
            if data != None and type(data) == pd.core.frame.DataFrame:
                self.data = data
            else:
                raise ValueError(
                    'The data type must be one of the '
                    'csv, json, sqlite3, or a DataFrame object.'
                )

        # same data column name and types
        self.db_tables_columns_types = dict()
        self.db_tables_columns_types[self.db_id] = dict()
        self.db_tables_columns_types[self.db_id][table_name] = dict()
        for col, _type in self.data.dtypes.items():
            # print(col, _type)
            if 'int' in str(_type).lower() or 'float' in str(_type).lower():
                _type = 'numeric'
            else:
                _type = 'categorical'
            self.db_tables_columns_types[self.db_id][table_name][col.lower()] = _type

        # convert all columns in data df to string lower case
        self.data.columns = self.data.columns.str.lower()

        # a dictionary of table column names
        self.db_tables_columns = {
            self.db_id:{
                self.table_id: list(self.data.columns)
            }
        }

        # saves the input data to a storage place in .'dataset/database
        # to be used by data processor
        if data_type == 'json' or data_type == 'sqlite3':
            # write to sqlite3 database
            dir = osp.join(self.temp_dataset_path, self.db_id)
            if not os.path.exists(dir):
                os.makedirs(dir)
            conn = sqlite3.connect(osp.join(dir, self.db_id+'.sqlite'))
            self.data.to_sql(self.table_id, conn, if_exists='replace', index=False)

        # create data processor and retrieve
        # all data from db and save to db_table_col_val_map
        self.data_processor = ProcessData4Training(db_url=self.temp_dataset_path)
        self.db_table_col_val_map = {}
        table_cols = self.data_processor.get_table_columns(self.db_id)
        self.db_table_col_val_map[self.db_id] = {}
        for table, cols in table_cols.items():
            col_val_map = self.data_processor.get_values_in_columns(
                self.db_id,
                table,
                cols,
                conditions='remove'
            )
            self.db_table_col_val_map[self.db_id][table] = col_val_map

    def show_dataset(self, top_rows=5):
        return self.data[:top_rows]