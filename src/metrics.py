from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from .data import write_json

def report(y,pred,labels,out,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ids=list(range(len(labels)))
    text=classification_report(y,pred,labels=ids,target_names=labels,zero_division=0)
    m={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=ids,average='macro',zero_division=0)),'classification_report':classification_report(y,pred,labels=ids,target_names=labels,output_dict=True,zero_division=0)}
    cm=confusion_matrix(y,pred,labels=ids)
    if len(labels)==2:
        tn,fp,fn,tp=cm.ravel();m.update(false_positive_rate=float(fp/max(tn+fp,1)),false_negative_rate=float(fn/max(tp+fn,1)),predicted_distress_fraction=float(np.mean(np.asarray(pred)==1)))
    write_json(out/f'{prefix}.json',m);(out/f'{prefix}.txt').write_text(text,encoding='utf-8')
    fig,ax=plt.subplots(figsize=(max(7,len(labels)*.32),max(6,len(labels)*.3)));ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,include_values=len(labels)<=10,xticks_rotation=90,colorbar=False);ax.set_title(prefix);fig.tight_layout();fig.savefig(out/f'{prefix}_confusion.png',dpi=130);plt.close(fig)
    print(prefix,{k:v for k,v in m.items() if k!='classification_report'});print(text);return m
