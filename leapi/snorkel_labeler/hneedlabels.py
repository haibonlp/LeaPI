
tag2id = {'Physiological': 0, 'Health': 1,  'Leisure':2 , 'Social':3,
                  'Finance': 4, 'Cognition':5, 'Mental': 6,  'None': 7 }
id2tag = { 0: 'Physiological', 1:'Health',  2:'Leisure', 3:'Social',
                       4: 'Finance', 5:'Cognition',  6:'Mental',  7:'None' }


def get_id_by_label(label):
    return tag2id[label]


def get_label_by_id(idx):
    return id2tag[idx]

def get_num_of_classes():
    return len(id2tag)


