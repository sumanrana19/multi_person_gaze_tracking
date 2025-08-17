import tkinter as tk
root = tk.Tk()
root.title("Test Tkinter")
root.geometry("300x100")
tk.Label(root, text="If you see this, Tkinter works!").pack()
root.mainloop()