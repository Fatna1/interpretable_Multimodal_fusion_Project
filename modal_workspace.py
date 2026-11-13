import modal
import subprocess 

app = modal.App("Multimodal-Fusion")

my_image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas", "requests", "matplotlib", "scipy", "h5py", )
    .add_local_dir(".", remote_path = "/Research_project")
)

@app.function(image = my_image, gpu = "T4")
def prepare_data():
    subprocess.run(["python", "/Research_project/prepare_data.py"], 
                   cwd = "/Research_project", 
                   check = True, 
                   )

    



