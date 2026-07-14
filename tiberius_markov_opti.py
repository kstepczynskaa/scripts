from pathlib import Path
import sys
import csv
import gc
import re
from subprocess import check_output, CalledProcessError
from types import SimpleNamespace

from skopt import gp_minimize
from skopt.space import Integer
from metrics import evaluate


# global paths
TIBERIUS_ROOT = Path("Tiberius").resolve()
sys.path.insert(0, str(TIBERIUS_ROOT))

from tiberius.main import run_tiberius


FASTA = Path("T4.fasta")
MODEL_CFG = Path("Tiberius/model_cfg/saccharomycota.yaml")
REFERENCE_GTF = Path("T4.gtf")
RESULTS_DIR = Path("results_final/saccharomycota")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "optimization_results.csv"

#  None, jeśli config ma default_seq_len,500004 tylko dla starych configów bez default_seq_len
SEQ_LEN = 500_004
MAX_CPU_TEMP = 85

def run_tiberius_direct(fasta_path: Path,output_gtf: Path,exon_len: int,intron_len: int,ir_len: int,):
    args = SimpleNamespace(genome=str(fasta_path),model_cfg=str(MODEL_CFG),out=str(output_gtf),

        hmm_initial_exon_len=exon_len,
        hmm_initial_intron_len=intron_len,
        hmm_initial_ir_len=ir_len,

        # pozostałe argumenty, które normalnie daje parser
        model=None,
        model_old="",
        model_lstm_old="",
        model_hmm="",

        seq_len=SEQ_LEN,
        batch_size=None,
        min_genome_seqlen=0,
        no_softmasking=False,

        clamsa=None,
        hmm_eps=0.01,
        parallel_factor=0,

        codingseq=None,
        protseq=None,
        group_size_limit=None,
    )

    run_tiberius(args)

    # sprzątanie po TensorFlow miedyz iter 
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()


def evaluate_prediction(reference_gtf: Path,predicted_gtf: Path,fasta: Path,):
    metrics, confusion_matrix = evaluate(reference_gtf,predicted_gtf,fasta,)

    return metrics, confusion_matrix


with RESULTS_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "iteration","exon_len","intron_len","ir_len",
        "macro_f1","mcc","ars","nmi","macro_precision","macro_recall",
        "intergenic_f1","intron_f1","cds_f1",
    ])


iteration = 0


def objective(params):
    global iteration
    iteration += 1

    exon_len, intron_len, ir_len = params

    print(
        f"START ITERATION {iteration}: "
        f"exon={exon_len}, "
        f"intron={intron_len}, "
        f"ir={ir_len}",
        flush=True,
    )

    if ir_len < 5:
        print(f"Skipping ir_len too small: {ir_len}", flush=True)
        return 1e6

    prediction_gtf = RESULTS_DIR / (
        f"prediction_"
        f"iter{iteration:03d}_"
        f"exon{exon_len}_"
        f"intron{intron_len}_"
        f"ir{ir_len}.gtf"
    )

    run_tiberius_direct(
        fasta_path=FASTA,
        output_gtf=prediction_gtf,
        exon_len=exon_len,
        intron_len=intron_len,
        ir_len=ir_len,
    )

    metrics, confusion_matrix = evaluate_prediction(
        reference_gtf=REFERENCE_GTF,
        predicted_gtf=prediction_gtf,
        fasta=FASTA,
    )
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
            iteration,exon_len,intron_len,ir_len,macro_f1,mcc,ars,nmi,macro_precision,macro_recall,intergenic_f1,intron_f1,cds_f1,
        ])

    print(
        f"[{iteration}] "
        f"F1={macro_f1:.4f} "
        f"MCC={mcc:.4f} "
        f"ARS={ars:.4f} "
        f"exon={exon_len} "
        f"intron={intron_len} "
        f"ir={ir_len}",
        flush=True
    )

    return -macro_f1


space = [
    Integer(50,900,name="hmm_initial_exon_len",),
    Integer(100,1100,name="hmm_initial_intron_len",),
    Integer(1,10,name="hmm_initial_ir_len",),]


res = gp_minimize(
    objective,space,
    n_calls=500,
    n_initial_points=50,
    random_state=42,
)


print()
print("BEST SCORE:")
print(-res.fun)

print()
print("BEST PARAMETERS:")
print({
    "hmm_initial_exon_len": res.x[0],
    "hmm_initial_intron_len": res.x[1],
    "hmm_initial_ir_len": res.x[2],
})

print()
print(f"All results saved to: {RESULTS_CSV}")
