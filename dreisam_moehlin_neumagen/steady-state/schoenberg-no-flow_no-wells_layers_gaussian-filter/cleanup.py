from pathlib import Path
import os
import click

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main")
def main(model_run):

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.dis.grb"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.lst"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.sto"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.cbc"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.npf"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.ic"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.rcha"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.chd"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.dis"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.tdis"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.oc"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.nam"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.hds"
        os.remove(file)
    except FileNotFoundError:
        pass

    try:
        base_path = Path(__file__).parent
        file = base_path / "output" / f"dmn_run_{model_run}.ims"
        os.remove(file)
    except FileNotFoundError:
        pass
    return


if __name__ == "__main__":
    main()
