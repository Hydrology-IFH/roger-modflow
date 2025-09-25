from pathlib import Path
import os
import click

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main")
def main(model_run):

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.dis.grb"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.lst"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.sto"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.cbc"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.npf"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.ic"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.rcha"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.chd"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.wel"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.dis"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.tdis"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.oc"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.nam"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.hds"
    os.remove(file)

    base_path = Path(__file__).parent
    file = base_path / "output" / f"dmn_run_{model_run}.ims"
    os.remove(file)
    return


if __name__ == "__main__":
    main()
