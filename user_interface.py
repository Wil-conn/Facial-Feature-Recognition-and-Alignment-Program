import tkinter
from tkinter import filedialog


class user_interface(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_window()
        self.selection = None
        self.dir_location = None

    def create_window(self):
        def sel():
            selection = "you selected " + str(var.get())
            self.selection = str(var.get())
            print(selection)

        def choose_dir():
            self.dir_location = filedialog.askdirectory()
            dir_label = tkinter.Label(self, text=self.dir_location).grid(row=3, column = 1)

        def run():
            self.quit()

        var = tkinter.StringVar()

        self.columnconfigure(0, minsize=250)
        self.columnconfigure(1, minsize=250)
        self.columnconfigure(2, minsize=250)

        feature_label = tkinter.Label(self, text="select which feature you would like to be aligned").grid(row=0, column=1)

        nose_button = tkinter.Radiobutton(self, text="nose", variable=var, value="nose", command=sel).grid(row=1, column=0)

        mouth_button = tkinter.Radiobutton(self, text="mouth", variable=var, value="mouth", command=sel).grid(row=1, column=1)

        eyes_button = tkinter.Radiobutton(self, text="eyes", variable=var, value="eyes", command=sel).grid(row=1, column=2)

        run_button = tkinter.Button(self, text="run", command=run).grid(row=4, column=1)

        dir_button = tkinter.Button(self, text="choose a directory of photos", command=choose_dir).grid(row=2, column=1)



    def say_test(self):
        print("test")


