copy of added cells to original 

```python

test_gff = RESULTS_DIR / "test_prediction_from_function.gff"

run_balrog_prediction(test_gff) #main function find genes wrapped inside 

metrics, confusion_matrix = evaluate(
    REFERENCE_GFF,
    test_gff,
    FASTA)


print(metrics)
print(confusion_matrix)

```

```python

def set_balrog_params(params):

    global max_gene_overlap
    global min_orf_length
    global score_threshold
    global weight_gene_prob
    global unidirectional_penalty_per_base
    global convergent_penalty_per_base
    global divergent_penalty_per_base

  

    max_gene_overlap = params["max_gene_overlap"]
    min_orf_length = params["min_orf_length"]
    score_threshold = params["score_threshold"]
    weight_gene_prob = params["weight_gene_prob"]
    unidirectional_penalty_per_base = params["unidirectional_penalty_per_base"]
    convergent_penalty_per_base = params["convergent_penalty_per_base"]
    divergent_penalty_per_base = params["divergent_penalty_per_base"]
```

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
import csv
import gc
```


```python 
iteration = 0

def objective(values):
    global iteration
    iteration += 1

    params = {
        "max_gene_overlap": values[0],
        "min_orf_length": values[1],
        "score_threshold": values[2],
        "weight_gene_prob": values[3],
        "unidirectional_penalty_per_base": values[4],
        "convergent_penalty_per_base": values[5],
        "divergent_penalty_per_base": values[6],}

  

    print()
    print(f"START ITERATION {iteration}")
    print(params, flush=True)

  

    set_balrog_params(params)
    predicted_gff = RESULTS_DIR / f"prediction_iter_{iteration:03d}.gff"
    run_balrog_prediction(predicted_gff)

    metrics, confusion_matrix = evaluate(
        REFERENCE_GFF,
        predicted_gff,
        FASTA)

    macro_f1 = metrics["Macro F1 Score"]
    mcc = metrics["Matthews Correlation Coefficient"]
    ars = metrics["Adjusted Rand Score"]
    nmi = metrics["Normalized Mutual Information Score"]
    macro_precision = metrics["Macro Precision"]
    macro_recall = metrics["Macro Recall"]
    intergenic_f1 = metrics["intergenic F1 Score"]
    intron_f1 = metrics["intron F1 Score"]
    cds_f1 = metrics["CDS F1 Score"]

    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            iteration,
            params["max_gene_overlap"],
            params["min_orf_length"],
            params["score_threshold"],
            params["weight_gene_prob"],
            params["unidirectional_penalty_per_base"],
            params["convergent_penalty_per_base"],
            params["divergent_penalty_per_base"],
            macro_f1,
            mcc,ars,
            nmi,
            macro_precision,
            macro_recall,
            intergenic_f1,intron_f1,cds_f1,])
    print(f"[{iteration}] "f"F1={macro_f1:.4f} "f"MCC={mcc:.4f} " f"ARS={ars:.4f}",flush=True,)
    gc.collect()
    return -macro_f1
    
    
space = [
    Integer(70, 110, name="max_gene_overlap"),
    Integer(39, 66, name="min_orf_length"),
    Real(0.36, 0.48, name="score_threshold"),
    Real(0.90, 1.10, name="weight_gene_prob"),
    Real(2.2, 3.4, name="unidirectional_penalty_per_base"),
    Real(3.0, 4.2, name="convergent_penalty_per_base"),
    Real(2.0, 3.2, name="divergent_penalty_per_base"),]
    
res = gp_minimize(objective,space,n_calls=500,n_initial_points=50,random_state=42,)


best_params = dict(zip([dim.name for dim in space], res.x))
print("BEST SCORE:")
print(-res.fun)

print("BEST PARAMETERS:")
print(best_params)

  

print("CSV:")
print(RESULTS_CSV)


files.download(str(RESULTS_CSV))
```