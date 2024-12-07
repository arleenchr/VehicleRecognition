import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # For displaying the selected image

def configure_grid(layout: tk.Tk, row: int, col: int, row_weight: list = None, col_weight: list = None):
  # Assume len(row_weight) = row and len(col_weight) = col
  for i in range(row):
    if (row_weight is not None):
      layout.rowconfigure(i, weight=row_weight[i])
    else:
      layout.rowconfigure(i, weight=1)

  for j in range(col):
    if (col_weight is not None):
      layout.columnconfigure(j, weight=col_weight[j])
    else:
      layout.columnconfigure(j, weight=1)

def select_input_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    filename = file_path.split("/")[-1]

    if file_path:
        # Display the selected file path
        selected_image_name.config(text=f"{filename}", fg="blue")
        
         # Load and display the image
        img = Image.open(file_path)
        img = img.resize((480,270))
        img_tk = ImageTk.PhotoImage(img)

        image_display_input_img.config(image=img_tk)
        image_display_input_img.image = img_tk  # Keep a reference to avoid garbage collection
        
    else:
        selected_image_name.config(text="No Selected Image", fg='red')



if __name__ == "__main__":
  # Create the main application window
  root = tk.Tk()
  root.title("Vehicle Recognition")
  root.geometry("1280x720")
  configure_grid(root, 1, 2, None, [1,3])

  ##############################################
  # Control panel
  panel = tk.Frame(root)
  panel.grid(row=0, column=0, sticky='nesw')
  configure_grid(panel, 10, 1)

  # Title
  title = tk.Label(panel, text="Vehicle Recognition", font=("Helvetica", 16, "bold"))
  title.grid(row=0, column=0, sticky='nsew')


  # Image Input Buttons
  image_input_grid = tk.Frame(panel)
  image_input_grid.grid(row=1, column=0, sticky='nsew')
  configure_grid(image_input_grid, 2, 2)
  # Top Left: Label
  image_input_label = tk.Label(image_input_grid, text="Input Image:", font=("Helvetica", 12), anchor="w")
  image_input_label.grid(row=0, column=0)
  # Top Right : Button
  button = tk.Button(image_input_grid, text="Select Image", command=select_input_image, font=("Helvetica", 12), anchor="w")
  button.grid(row=0, column=1)
  # Bottom Left: Selected Label
  selected_image_label = tk.Label(image_input_grid, text="Selected Image:", font=("Helvetica", 12), anchor="w")
  selected_image_label.grid(row=1, column=0)
  # Bottom Right: Selected Name
  selected_image_name = tk.Label(image_input_grid, text="No Selected Image", font=("Helvetica", 12), fg="red", anchor="w")
  selected_image_name.grid(row=1, column=1)




  ######################################################
  # Image display
  img_display_grid = tk.Frame(root, bg='lightblue')
  img_display_grid.grid(row=0, column=1, sticky='nsew')
  configure_grid(img_display_grid, 2, 1)

  # Display Input
  image_display_input_grid = tk.Frame(img_display_grid)
  image_display_input_grid.grid(row=0, column=0)
  configure_grid(image_display_input_grid, 2, 1, [4, 1], None)
  image_display_input_label = tk.Label(image_display_input_grid, text="Input Image Display", font=("Helvetica", 12), anchor="w")
  image_display_input_label.grid(row=0, column=0)
  image_display_input_img = tk.Label(image_display_input_grid)
  image_display_input_img.grid(row=1, column=0)


  

  # Display Result
  image_display_result_grid = tk.Frame(img_display_grid)
  image_display_result_grid.grid(row=1, column=0)

  # Run the application
  root.mainloop()