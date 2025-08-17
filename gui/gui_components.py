import tkinter as tk
from tkinter import messagebox, Toplevel, Listbox, Button, Label, Frame
from datetime import datetime

def show_popup(message):
    """Show a popup message"""
    root = tk.Tk()
    root.withdraw()  
    messagebox.showinfo("Info", message)
    root.destroy()

def show_error(message):
    """Show an error popup"""
    root = tk.Tk()
    root.withdraw()  
    messagebox.showerror("Error", message)
    root.destroy()

def select_people(attendance_df):
    """GUI for selecting exactly TWO people from attendance list"""
    print("[DEBUG][GUI] select_people called. attendance_df:")
    print(attendance_df)
    root = tk.Tk()
    # root.withdraw()  # Temporarily comment out to force main window to show
    top = Toplevel(root)
    top.title("Select TWO People for Gaze Tracking")
    top.geometry("400x300")
    top.resizable(False, False)

    # Instructions label
    instruction_label = Label(top, text="Please select exactly TWO people:", 
                            font=('Arial', 12, 'bold'))
    instruction_label.pack(pady=10)

    # Listbox with multiple selection enabled
    listbox = Listbox(top, selectmode='multiple', font=('Arial', 10))
    listbox.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Add all unique names to listbox
    for name in attendance_df["Name"].unique():
        listbox.insert("end", name)

    selected_names = []
   
    def on_select():
        nonlocal selected_names
        selected_indices = listbox.curselection()
        
        if len(selected_indices) != 2:
            messagebox.showerror("Selection Error", 
                               "Please select exactly TWO people from the list.")
            return
        
        selected_names = [listbox.get(idx) for idx in selected_indices]
        top.destroy()
    
    def on_cancel():
        top.destroy()
    
    # Buttons
    button_frame = Frame(top)
    button_frame.pack(pady=10)
    
    select_button = Button(button_frame, text="Select", command=on_select, 
                          bg='green', fg='white', width=10)
    select_button.pack(side=tk.LEFT, padx=5)
    
    cancel_button = Button(button_frame, text="Cancel", command=on_cancel, 
                          bg='red', fg='white', width=10)
    cancel_button.pack(side=tk.LEFT, padx=5)
   
    # Center the window on screen
    top.transient(root)
    top.grab_set()
    top.wait_window()
    
    return selected_names

def show_attentiveness_report(student_name, percent, productivity):
    """Display attentiveness report in a GUI"""
    root = tk.Tk()
    root.title(f"Attentiveness Report - {student_name}")
    root.geometry("500x300")
    root.resizable(False, False)
    
    report_text = f"""
    ATTENTIVENESS REPORT FOR {student_name.upper()}
    {'='*40}
    Tracking Date: {datetime.now().date()}
    Attentiveness Percentage: {percent:.2f}%
    Productivity Assessment: {productivity}
    
    Analysis:
    - Scores above 70% indicate high focus
    - Scores between 40-70% indicate moderate focus
    - Scores below 40% indicate poor focus
    """
    
    text_widget = tk.Text(root, wrap=tk.WORD, padx=10, pady=10, 
                         font=('Arial', 12), state='normal')
    text_widget.insert(tk.END, report_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(expand=True, fill=tk.BOTH)
    
    close_btn = Button(root, text="Close", command=root.destroy, 
                      bg='blue', fg='white', width=15)
    close_btn.pack(pady=10)
    
    # Center the window on screen
    root.transient()
    root.grab_set()
    root.mainloop()

def confirm_action(message, title="Confirm"):
    """Show confirmation dialog"""
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, message)
    root.destroy()
    return result

def get_user_input(prompt, title="Input"):
    """Get user input via dialog"""
    root = tk.Tk()
    root.withdraw()
    from tkinter import simpledialog
    result = simpledialog.askstring(title, prompt)
    root.destroy()
    return result