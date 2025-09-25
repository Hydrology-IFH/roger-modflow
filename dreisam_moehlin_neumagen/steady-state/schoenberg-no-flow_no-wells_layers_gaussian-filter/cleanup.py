from pathlib import Path
import os
import click

@click.option("-mr", "--model-run", type=int, default=5)
@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent / "output" )
@click.command("main")
def main(model_run, tmp_dir):

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.dis.grb"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.lst"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.sto"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.cbc"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.npf"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.ic"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.rcha"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.chd"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.dis"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.tdis"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.oc"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.nam"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.hds"
    os.remove(file)

    
    file = Path(tmp_dir) / f"dmn_run_{model_run}.ims"
    os.remove(file)
    return


if __name__ == "__main__":
    main()
