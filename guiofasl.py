import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
from keras.models import load_model
from gtts import gTTS

import os
language = 'en'
# Create a Tkinter window
root = tk.Tk()
x=0
root.title("Image Viewer")
mod=load_model(r'ASL-recognizer.h5')
# Create a function to open a file dialog and get the file path
def open_file():
    file_path = filedialog.askopenfilename(title="Select Image",
                                           filetypes=(("JPEG files", "*.jpeg"),
                                                      ("PNG files", "*.png"),
                                                      ("All files", ".")))
    # Load the selected image using matplotlib
    image = plt.imread(file_path)

    # Display the image using a matplotlib figure
    fig = plt.figure(figsize=(5, 5))
    img=image
    image=cv2.resize(image,(256,256))
    image=np.array(image,dtype='float32')/255.0
    image=np.reshape(image,(1,256,256,3))
    p=mod.predict(image)
    ans=np.argmax(p)
    x=ans
    if x>9:
      print(chr(87+x))
      mytext = chr(87+x)
    else:
      print(x)
      mytext= str(x)
    plt.imshow(img)
    plt.title(mytext)
    plt.axis("off")
    plt.show()
    myobj = gTTS(text=mytext, lang=language, slow=False)

    myobj.save("welcome.mp3")
      
  
    os.system("welcome.mp3")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # Display the file path where the image has been stored
    file_label.config(text="File: {}".format(file_path))

# Create a button to open the file dialog
open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack(pady=10)

# Create a label to display the file path
file_label = tk.Label(root, text="")
file_label.pack()

# Run the Tkinter event loop
root.mainloop()

