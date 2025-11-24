import torch
import torch.utils.data as tud
import pandas as pd
import os
import random
from dataclasses import dataclass
from SERSome_AD_wen_coeff.models import GPTConfig

spec_modal_map = {'negative-R2-638nm': 0, 'negative-R1-638nm': 1, 'negative-R1-532nm': 2,
                       'negative-R2-532nm': 3, 'positive-R1-532nm': 4, }

class Dataset(tud.Dataset):
    def __init__(self, type = 'train',seg_seed = 0,config: GPTConfig = None):
        self.spectra_dir = 'spectra/'
        self.spectra_meta_dir = 'spectra_meta/'
        self.all_spectra_path()
        self.all_spectra_meta_path()
        self.patient2idx, self.idx2patient = load_double_blind_mapping()
        self.ehr = self.load_ehr('EHR_revised_4.xlsx')
        self.load_gene('Gene.xlsx')

        self.type = type
        self.seg_dataset(seg_seed = seg_seed)
        self.config = config
        self.spec_meta_dim = 27

        if self.config.if_data_balance:
            self.label_balance()
        if self.type == 'train':
            self.idx = self.train_idx
        elif self.type == 'val':
            self.idx = self.val_idx
        elif self.type == 'test':
            self.idx = self.test_idx

        # print('mapping number:',len(self.idx_mapping.keys()))
        print('Total number of patients:', len(self.train_idx) + len(self.val_idx) + len(self.test_idx))
        # print('Number of patients in train:', len(self.train_idx))
        # print('Number of patients in val:', len(self.val_idx))
        # print('Number of patients in test:', len(self.test_idx))


    def gene_mapping(self,gene):
        if gene == 'TT':
            return 0
        elif gene == 'CT':
            return 1
        elif gene == 'CC':
            return 2
        elif gene == 'TC':
            return 3
        else:
            return 4

    def AD_ralation_mapping(self,relation):
        if 'E2' in relation:
            return 0
        elif 'E3' in relation:
            return 1
        elif 'E4' in relation:
            return 2
        else:
            return 3

    def label_mapping(self,label):
        if label == 'AD' or 'ad' in label:
            return 0
        elif label == 'MCI':
            return 1
        elif label == 'Control':
            return 2
        else:
            return 3

    def age_normalization(self,age):
        return (age - 40) / 20

    def all_spectra_path(self):
        # mapping the idx to the spectra path
        spectra_pth = all_files_path(self.spectra_dir)
        idx_mapping = dict()
        for pth in spectra_pth:

            idx = int(pth.split('/')[-1].split('-')[0])
            if idx not in idx_mapping.keys():
                idx_mapping[idx] = [pth]
            else:
                idx_mapping[idx].append(pth)
        for idx in idx_mapping.keys():
            idx_mapping[idx] = list(set(idx_mapping[idx]))
        self.idx_mapping = idx_mapping
        # print('spec number', len(idx_mapping.keys()))

    def all_spectra_meta_path(self):
        # mapping the idx to the spectra path
        spectra_pth = all_files_path(self.spectra_meta_dir)
        idx_mapping = dict()
        for pth in spectra_pth:
            idx = int(pth.split('/')[-1].split('-')[0])
            if idx not in idx_mapping.keys():
                idx_mapping[idx] = [pth]
            else:
                idx_mapping[idx].append(pth)
        for idx in idx_mapping.keys():
            idx_mapping[idx] = list(set(idx_mapping[idx]))
        self.meta_mapping = idx_mapping
        # print('meta number',len(idx_mapping.keys()))

    def tokenizer(self,text, pad_len = 6, if_pad = True):
        meta_vocab = {'海马': 0, '两侧': 1, '左侧': 2, '右侧': 3, '萎缩': 4, 'MTA=1': 5,
                      'MTA=2': 6, 'MTA=3': 7,'脑内散在腔梗': 8, '脑内多发腔梗': 9,
                      '脑内少许腔梗': 10, '老年脑': 11, '脑白质疏松': 12,
                      'Fazekas2': 13, 'Fazekas3': 14, 'Fazekas1': 15,
                      '脑萎缩': 16, '缺血': 17, '未见明显异常': 18, 'nan': 19}
        token = []
        text = str(text)
        # print(text)
        for meta_word in meta_vocab:
            if meta_word in text:
                token.append(meta_vocab[meta_word])
        if if_pad:
            if len(token) > pad_len:
                token = token[:pad_len]
            else:
                token += [19] * (pad_len - len(token))
        token = torch.tensor(token)
        return token

    def load_gene(self,gene_path):
        gene = pd.read_excel(gene_path)
        gene = gene.fillna(999)
        self.gene_APOE_R176 = dict()
        self.gene_APOE_C130 = dict()
        self.genetic_relation = dict()
        for idx, row in gene.iterrows():
            patient_id = int(row['原始编号'].split('D')[-1])
            self.gene_APOE_C130[patient_id] = self.gene_mapping(row['APOE:C130 （rs429358）'])
            self.gene_APOE_R176[patient_id] = self.gene_mapping(row['APOE:R176 （rs7412）'])
            self.genetic_relation[patient_id] = self.AD_ralation_mapping(row['AD relation'])
        # print(self.gene_APOE_C130,self.gene_APOE_R176)
        print('gene number:',len(self.gene_APOE_C130.keys()))

    def load_ehr(self, ehr_path):
        '''
            row[0] is the patient id
            row[1] is the label
            row[2:] is the ehr data
            row[2] 痴呆家族史（1=有；0=无）
            row[3] MRI
            row[4] 年龄
            row[5] 性别（男1女0）
            row[6] 教育程度（0=文盲，1=小学，2=初中，3=高中，4=大学）
            row[7] 教育时长（年）
            row[8] HAMA
            row[9] HADA
            row[10] MMSE
            row[11] 定向力（10）
            row[12] 词语即刻记忆（3）
            row[13] 注意力及计算（5）
            row[14] 词语回忆（3）
            row[15] 语言能力
            row[16] 结构模仿
            row[17] MoCA
            row[18] 视空间与执行能力【交替连线检验+视空间（立方体）+视空间（钟表）】
            row[19] 交替连线实验
            row[20] 视空间（立方体）
            row[21] 视空间（钟表）
            row[22] 命名
            row[23] 记忆1（不计分）
            row[24] 记忆2（不计分）
            row[25] 注意力（顺背1-倒背1-警觉性1-计算3）
            row[26] 顺背
            row[27] 倒背
            row[28] 警觉性
            row[29] 计算
            row[30] 句子复述（2分）
            row[31] 词语流畅（1分）
            row[32] 抽象
            row[33] 延迟记忆
            row[34] 延迟2
            row[35] 延迟3
            row[36] 定向
            row[37] 白细胞绝对值(X10*9/L)
            row[38] 中性粒百分数N%
            row[39] 淋巴细胞百分数L%
            row[40] 红细胞（X10*12/L)
            row[41] 血红蛋白
            row[42] 血小板计数
            row[43] TBIL
            row[44] 总蛋白
            row[45] 白蛋白
            row[46] ALT
            row[47] AST
            row[48] 尿素
            row[49] 肌酐（酶法）
            row[50] 尿素肌酐比值
            row[51] eGFR
            row[52] 尿酸
            row[53] TC
            row[54] TG
            row[55] HDL
            row[56] LDL
            row[57] 同型半胱氨酸
            row[58] 甲状腺素
            row[59] 三碘甲腺原氨酸
            row[60] 促甲状腺素
            row[61] FT4(?指游离甲状腺素)
            row[62] FT3

        '''
        patinet2idx, idx2patinet = load_double_blind_mapping()
        ehr = pd.read_excel(ehr_path,keep_default_na=True)
        # replace nan as 999
        ehr = ehr.fillna(999)
        # replace x as 999
        ehr = ehr.replace('x',999)
        ehr = ehr.replace('X',999)
        ehr = ehr.replace('<5', 999)
        header = ehr.columns
        self.label = dict()
        self.family_history = dict()
        self.MRI = dict()
        self.age = dict()
        self.gender = dict()
        self.education = dict()
        self.education_duration = dict()
        self.HAMA = dict()
        self.HADA = dict()
        self.MMSE = dict()
        self.orientation = dict()
        self.immediate_memory = dict()
        self.attention = dict()
        self.delayed_recall = dict()
        self.language = dict()
        self.structure_imitation = dict()
        self.MoCA = dict()
        self.visual_space_execution = dict()
        self.alternate_line_test = dict()
        self.visual_space_cube = dict()
        self.visual_space_clock = dict()
        self.naming = dict()
        self.memory1 = dict()
        self.memory2 = dict()
        self.attention2 = dict()
        self.forward = dict()
        self.backward = dict()
        self.alertness = dict()
        self.calculation = dict()
        self.sentence_repetition = dict()
        self.word_fluency = dict()
        self.abstract = dict()
        self.delayed_memory = dict()
        self.delayed_memory2 = dict()
        self.delayed_memory3 = dict()
        self.orientation = dict()
        self.white_blood_cell = dict()
        self.neutrophil_percentage = dict()
        self.lymphocyte_percentage = dict()
        self.red_blood_cell = dict()
        self.hemoglobin = dict()
        self.platelet_count = dict()
        self.TBIL = dict()
        self.total_protein = dict()
        self.albumin = dict()
        self.ALT = dict()
        self.AST = dict()
        self.urea = dict()
        self.creatinine = dict()
        self.urea_creatinine_ratio = dict()
        self.eGFR = dict()
        self.uric_acid = dict()
        self.TC = dict()
        self.TG = dict()
        self.HDL = dict()
        self.LDL = dict()
        self.homocysteine = dict()
        self.thyroxine = dict()
        self.thyroxine_triiodothyronine = dict()
        self.thyroid_stimulating_hormone = dict()
        self.FT4 = dict()
        self.FT3 = dict()
        self.protein = dict()

        # print('ehr number:',len(ehr))
        # print(self.idx_mapping.keys())
        for idx, row in ehr.iterrows():
            patient_id = int(row[header[0]])
            if patient_id in idx2patinet.keys():
                idx = idx2patinet[patient_id]

                if idx not in self.idx_mapping.keys():
                    # print('idx not in the spectra:',idx)
                    # print('patient_id:',patient_id)
                    # self.label[idx] = row[header[1]]
                    continue
                self.regisiter(idx, row, header)


        # print(self.protein)

    def seg_dataset(self,ratio=[0.8,0.1,0.1],seg_seed=0):
        #
        torch.manual_seed(seg_seed)
        idxs = torch.tensor(list(self.idx_mapping.keys()))
        # idxs = torch.randperm(len(idxs))

        label_count = {
            'AD':[],
            'MCI':[],
            'Control':[],
            # 'VCI':[],
            'Other':[]
        }
        # print(self.label.keys())
        # print(idxs)
        for idx in idxs:
            if int(idx) in self.label.keys():
                if self.label[int(idx)] not in label_count.keys():
                    label_count['Other'].append(int(idx))
                else:
                    label_count[self.label[int(idx)]].append(int(idx))
        # print('total number of patients:', len(idxs),
        #       len(label_count['AD'])+len(label_count['MCI'])+len(label_count['Control'])+len(label_count['Other']))
        #set the seed
        for key in label_count.keys():
            random.seed(seg_seed)
            random.shuffle(label_count[key])
            #seg the dataset
            train_size = int(len(label_count[key]) * ratio[0])
            val_size = int(len(label_count[key]) * ratio[1])
            test_size = len(label_count[key]) - train_size - val_size
            label_count[key] = {
                'train':label_count[key][:train_size],
                'val':label_count[key][train_size:train_size + val_size],
                'test':label_count[key][train_size + val_size:]
            }
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        for key in label_count.keys():
            self.train_idx += label_count[key]['train']
            self.val_idx += label_count[key]['val']
            self.test_idx += label_count[key]['test']

        # idxs = torch.randperm(len(idxs))
        # train_size = int(len(idxs) * ratio[0])
        # val_size = int(len(idxs) * ratio[1])
        # test_size = len(idxs) - train_size - val_size
        # self.train_idx = idxs[:train_size]
        # self.val_idx = idxs[train_size:train_size + val_size]
        # self.test_idx = idxs[train_size + val_size:]
        # print('train size:',len(self.train_idx))
        # print('val size:',len(self.val_idx))
        # print('test size:',len(self.test_idx))

    def __len__(self):

        # label distribution in the dataset
        # print('label distribution in the dataset')
        label_count = {
            'AD':[],
            'MCI':[],
            'Control':[],
            # 'VCI':[],
            'Other':[]
        }
        for idx in self.train_idx:
            if int(idx) in self.label.keys():
                if self.label[int(idx)] not in label_count.keys():
                    label_count['Other'].append(int(idx))
                else:
                    label_count[self.label[int(idx)]].append(int(idx))
        # print('train label distribution:')
        # for key in label_count.keys():
        #     print(key,':',len(label_count[key]))

        label_count = {
            'AD':[],
            'MCI':[],
            'Control':[],
            # 'VCI':[],
            'Other':[]
        }
        for idx in self.val_idx:
            if int(idx) in self.label.keys():
                if self.label[int(idx)] not in label_count.keys():
                    label_count['Other'].append(int(idx))
                else:
                    label_count[self.label[int(idx)]].append(int(idx))
        # print('val label distribution:')
        # for key in label_count.keys():
        #     print(key,':',len(label_count[key]))
        label_count = {
            'AD':[],
            'MCI':[],
            'Control':[],
            # 'VCI':[],
            'Other':[]
        }
        for idx in self.test_idx:
            if int(idx) in self.label.keys():
                if self.label[int(idx)] not in label_count.keys():
                    label_count['Other'].append(int(idx))
                else:
                    label_count[self.label[int(idx)]].append(int(idx))
        # print('test label distribution:')
        # for key in label_count.keys():
        #     print(key,':',len(label_count[key]))


        if self.type == 'train':
            return len(self.train_idx)
        elif self.type == 'val':
            return len(self.val_idx)
        elif self.type == 'test':
            return len(self.test_idx)
        else:
            raise ValueError('type must be train,val or test')

    def label_balance(self):
        # 统计label的分布
        label_count = {
            'AD':[],
            'MCI':[],
            'Control':[],
            # 'VCI':[],
            'Other':[]
        }

        for idx in self.train_idx:
            if int(idx) in self.label.keys():
                if self.label[int(idx)] not in label_count.keys():
                    label_count['Other'].append(int(idx))
                else:
                    label_count[self.label[int(idx)]].append(int(idx))
        original_label_count = label_count.copy()
        # balance the label
        max_len = max([len(label_count['AD']),len(label_count['MCI']),len(label_count['Control'])])

        for key in label_count.keys():
            if len(label_count[key]) < max_len:
                label_count[key] = label_count[key] * (max_len // len(label_count[key]))
                label_count[key] += label_count[key][:max_len % len(label_count[key])]
            else:
                label_count[key] = label_count[key][:max_len]
        # print(label_count)
        self.train_idx = label_count['AD'] + label_count['MCI'] + label_count['Control'] \
                         + original_label_count['Other']\
                         # + original_label_count['VCI']
        self.train_idx = torch.tensor(self.train_idx)




    def __getitem__(self, idx):
        '''

        return:
            info_dict: a dictionary containing the information of the patient
            label: the label of the patient
            MRI: the MRI data of the patient
            age: the age of the patient
            gender: the gender of the patient
            education: the education of the patient
            education_duration: the education duration of the patient
            MMSE: the MMSE score of the patient
            moca: the MoCA score of the patient
            neurological_test: the neurological test of the patient
            blood_test: the blood test of the patient


        '''

        if self.type == 'train':
            idx = self.train_idx[idx]
        elif self.type == 'val':
            idx = self.val_idx[idx]
        elif self.type == 'test':
            idx = self.test_idx[idx]

        info = self.get_info(int(idx))
        if info is not None:
            return info
        else:
            return self.__getitem__(random.randint(0,self.__len__()-1))

    def regisiter(self,idx,row,header):
        self.label[idx] = row[header[1]]
        self.family_history[idx] = int(row[header[2]])
        self.MRI[idx] = self.tokenizer(row[header[3]])
        self.age[idx] = self.age_normalization(row[header[4]])
        self.gender[idx] = int(row[header[5]])
        self.education[idx] = int(row[header[6]])
        self.education_duration[idx] = row[header[7]]
        self.HAMA[idx] = row[header[8]]
        self.HADA[idx] = row[header[9]]
        self.MMSE[idx] = row[header[10]]
        self.orientation[idx] = row[header[11]]
        self.immediate_memory[idx] = row[header[12]]
        self.attention[idx] = row[header[13]]
        self.delayed_recall[idx] = row[header[14]]
        self.language[idx] = row[header[15]]
        self.structure_imitation[idx] = row[header[16]]
        self.MoCA[idx] = row[header[17]]
        self.visual_space_execution[idx] = row[header[18]]
        self.alternate_line_test[idx] = row[header[19]]
        self.visual_space_cube[idx] = row[header[20]]
        self.visual_space_clock[idx] = row[header[21]]
        self.naming[idx] = row[header[22]]
        self.memory1[idx] = row[header[23]]
        self.memory2[idx] = row[header[24]]
        self.attention2[idx] = row[header[25]]
        self.forward[idx] = row[header[26]]
        self.backward[idx] = row[header[27]]
        self.alertness[idx] = row[header[28]]
        self.calculation[idx] = row[header[29]]
        self.sentence_repetition[idx] = row[header[30]]
        self.word_fluency[idx] = row[header[31]]
        self.abstract[idx] = row[header[32]]
        self.delayed_memory[idx] = row[header[33]]
        self.delayed_memory2[idx] = row[header[34]]
        self.delayed_memory3[idx] = row[header[35]]
        self.orientation[idx] = row[header[36]]
        self.white_blood_cell[idx] = row[header[37]]
        self.neutrophil_percentage[idx] = row[header[38]]
        self.lymphocyte_percentage[idx] = row[header[39]]
        self.red_blood_cell[idx] = row[header[40]]
        self.hemoglobin[idx] = row[header[41]]
        self.platelet_count[idx] = row[header[42]]
        self.TBIL[idx] = row[header[43]]
        self.total_protein[idx] = row[header[44]]
        self.albumin[idx] = row[header[45]]
        self.ALT[idx] = row[header[46]]
        self.AST[idx]= row[header[47]]
        self.urea[idx] = row[header[48]]
        self.creatinine[idx] = row[header[49]]
        self.urea_creatinine_ratio[idx] = row[header[50]]
        self.eGFR[idx] = row[header[51]]
        self.uric_acid[idx] = row[header[52]]
        self.TC[idx] = row[header[53]]
        self.TG[idx] = row[header[54]]
        self.HDL[idx] = row[header[55]]
        self.LDL[idx] = row[header[56]]
        self.homocysteine[idx] = row[header[57]]
        self.thyroxine[idx] = row[header[58]]
        self.thyroxine_triiodothyronine[idx] = row[header[59]]
        self.thyroid_stimulating_hormone[idx] = row[header[60]]
        self.FT4[idx] = row[header[61]]
        self.FT3[idx] = row[header[62]]
        self.protein[idx] = row[header[63]]

    def get_info(self,idx):
        if idx in self.idx_mapping.keys() and idx in self.label.keys():

            info_dict ={
                'label':self.label_mapping(self.label[idx])}

            if self.config.apply_mri:
                info_dict['MRI'] = self.MRI[idx]

            if self.config.apply_ehr:
                info_dict.update({
                'family_history':self.family_history[idx],
                'age':self.age[idx],
                'gender':self.gender[idx],
                'education':self.education[idx],
                'education_duration':self.education_duration[idx],
                })

            if self.config.apply_protein:
                info_dict['protein'] = self.protein[idx]

            if self.config.apply_neurological_test:
                info_dict.update({
                'MMSE':self.MMSE[idx],
                'moca':self.MoCA[idx],
                'neurological_test':[
                    self.orientation[idx],
                    self.immediate_memory[idx],
                    self.attention[idx],
                    self.delayed_recall[idx],
                    self.language[idx],
                    self.structure_imitation[idx],
                    self.visual_space_execution[idx],
                    self.alternate_line_test[idx],
                    self.visual_space_cube[idx],
                    self.visual_space_clock[idx],
                    self.naming[idx],
                    self.memory1[idx],
                    self.memory2[idx],
                    self.attention2[idx],
                    self.forward[idx],
                    self.backward[idx],
                    self.alertness[idx],
                    self.calculation[idx],
                    self.sentence_repetition[idx],
                    self.word_fluency[idx],
                    self.abstract[idx],
                    self.delayed_memory[idx],
                    self.delayed_memory2[idx],
                    self.delayed_memory3[idx],
                    self.orientation[idx]
                ]})

            if self.config.apply_blood_test:
                info_dict.update({
                'blood_test':[
                    self.white_blood_cell[idx],
                    self.neutrophil_percentage[idx],
                    self.lymphocyte_percentage[idx],
                    self.red_blood_cell[idx],
                    self.hemoglobin[idx],
                    self.platelet_count[idx],
                    self.TBIL[idx],
                    self.total_protein[idx],
                    self.albumin[idx],
                    self.ALT[idx],
                    self.AST[idx],
                    self.urea[idx],
                    self.creatinine[idx],
                    self.urea_creatinine_ratio[idx],
                    self.eGFR[idx],
                    self.uric_acid[idx],
                    self.TC[idx],
                    self.TG[idx],
                    self.HDL[idx],
                    self.LDL[idx],
                    self.homocysteine[idx],
                    self.thyroxine[idx],
                    self.thyroxine_triiodothyronine[idx],
                    self.thyroid_stimulating_hormone[idx],
                    self.FT4[idx],
                    self.FT3[idx]
                ]
            })
            if self.config.apply_gene:
                # print(self.gene_APOE_C130)
                gene_idx = self.patient2idx[idx]
                if gene_idx not in self.gene_APOE_C130.keys():
                    info_dict['gene_c130'] = 999
                    info_dict['gene_r176'] = 999
                    info_dict['genetic_relation'] = 999
                else:
                    info_dict['gene_c130'] = self.gene_APOE_C130[gene_idx]
                    info_dict['gene_r176'] = self.gene_APOE_R176[gene_idx]
                    info_dict['genetic_relation'] = self.genetic_relation[gene_idx]

            if self.config.apply_spec:
                pth = self.idx_mapping[idx]
                info_dict['spec'] = torch.zeros(201*5,self.config.signal_length)
                # info_dict['spec_name'] = torch.zeros(5)
                for idx,p in enumerate(pth):
                    # print(p)
                    spec = torch.load(p,weights_only=False)
                    spec = torch.nn.functional.interpolate(spec.unsqueeze(0),
                            size=self.config.signal_length,mode='linear').squeeze(0)
                    info_dict['spec'][idx*201:(idx+1)*201] = spec
                    # info_dict['spec_name'][idx] = get_id(p,spec_modal_map)
                    if idx == 4:
                        break

            if self.config.apply_spec_meta:
                if idx not in self.meta_mapping.keys():
                    # print('no meta data for patient:',idx)
                    info_dict['spec_meta'] = torch.zeros(self.spec_meta_dim,201*2).float()
                else:
                    pth = self.meta_mapping[idx]
                    info_dict['spec_meta'] = torch.zeros(self.spec_meta_dim,201*2).float()
                    for idx,p in enumerate(pth):
                        meta = torch.load(p)
                        # meta.shape = (201,27)
                        info_dict['spec_meta'][:,idx*201: (idx+1)*201] = meta
                        if idx == 2:
                            break

                # print(info_dict)
            return info_dict



def load_double_blind_mapping():
    '''
    Load the double blind mapping from the excel file
    Returns:
        patinet2idx: a dictionary mapping from patient id to index
        idx2patinet: a dictionary mapping from index to patient id

    '''
    double_blind_mapping = pd.read_excel('Double_list.xlsx')
    header = double_blind_mapping.columns
    # print(double_blind_mapping.values)
    patinet2idx = {}
    idx2patinet = {}
    for idx, row in double_blind_mapping.iterrows():
        patinet2idx[row[header[0]]] = row[header[1]]
        idx2patinet[row[header[1]]] = row[header[0]]
    # print('Mapping number of patients:', len(patinet2idx.keys()))
    # print(idx2patinet.keys())
    return patinet2idx, idx2patinet



def all_files_path(rootDir):
    filepaths = []
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            file_path = os.path.join(root, file)
            filepaths.append(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            subdir_pth = all_files_path(dir_path)
            filepaths.extend(subdir_pth)
    return filepaths


def get_id(pth_name:str,map_dict:dict):
    for (key,value) in map_dict.items():
        if key in pth_name:
            return value

    return 6

if __name__ == '__main__':
    @dataclass
    class GPTConfig:
        block_size: int = 64
        n_layer: int = 4
        n_head: int = 16
        n_embd: int = 64
        dropout: float = 0.1
        bias: bool = True
        cls_num: int = 2
        apply_ehr: bool = True
        apply_gene: bool = True
        apply_mri: bool =True
        apply_blood_test: bool = True
        apply_neurological_test: bool = True
        apply_spec: bool = True
        apply_protein: bool = True
        if_data_balance: bool = True
        signal_length: int = 512
        device: str = 'cuda:0'

    dataset = Dataset(config=GPTConfig())
    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])
        for key in dataset[i].keys():
            if str in [type(dataset[i][key])]:
                print(key,dataset[i][key])

    loader = tud.DataLoader(dataset, batch_size=4, shuffle=True)
    dataset.load_gene('Gene.xlsx')
