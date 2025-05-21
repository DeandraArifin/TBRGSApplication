import pandas as pd
import tkinter as tk
from tkintermapview import TkinterMapView
from pathfinder import run_model, load
from functools import partial

drawn_paths = []
drawn_path_times = []

scalers = models = hourly = None  
        
def on_model_select(event, model_opt):
    global scalers, models, hourly
    model_name = model_opt.get()
    scalers, models, hourly = load(model_name)
    print("Models loaded")

def draw_markers(map_widget, scats_site_data):
   for _, row in scats_site_data.iterrows():
        map_widget.set_marker(
            row["NB_LATITUDE"],
            row["NB_LONGITUDE"],
            text=f"{row['SCATS Number']}"
        )
         
def draw_route(map_widget, path_coords, colour):
    
    if not path_coords:
        print("No valid coordinates for this path!")
        
    line = map_widget.set_path(path_coords, color=colour)
    
    lats = [coord[0] for coord in path_coords]
    lons = [coord[1] for coord in path_coords]

    #calculate bounding box
    top_left = (max(lats), min(lons))      
    bottom_right = (min(lats), max(lons))  

    #zoom map to drawn paths
    map_widget.fit_bounding_box(top_left, bottom_right)
    return line

def on_find_route(left_frame, map_widget, origin_opt, dest_opt, time_opt, model_opt, scats_site_data, error_text, toggle_frame, travel_time_label):
    
    global drawn_paths
    global drawn_path_times
    
    origin = origin_opt.get()
    destination = dest_opt.get()
    time = time_opt.get()
    model = model_opt.get()
    
    print("ORIGIN:", origin)
    print("DESTINATION:", destination)
    print("TIME:", time)
    print("MODEL:", model)

#just to make sure all labels are empty at the start
    if error_text.winfo_ismapped():
        error_text.pack_forget()
    
    for path in drawn_paths:
        path['line'].delete()
    drawn_paths.clear()
    
    for label in drawn_path_times:
        label.pack_forget()
    drawn_path_times.clear()
    
    for widget in toggle_frame.winfo_children():
        widget.destroy()
    
    if not origin or not destination or not time or not model:
        if not error_text.winfo_ismapped():  # Check if the label is already packed
            error_text.pack(anchor='w', pady=30)
        print("All fields must be selected")
        return
    
    print(f"Origin: {origin}, Destination: {destination}, Time: {time}, Model: {model}")
    
    colours = ['red', 'blue', 'green', 'purple', 'yellow']
    paths, pathtimes = run_model(origin, destination, time, model, scalers, models, hourly)
    print(f"Returned paths: {paths}")
    
    paths = sorted(paths, key=len, reverse=True)
    idx = 0
    
    if paths:
        for idx, path in enumerate(paths):
            path_coords = []
            if str(path[0]) != str(origin):
                path.insert(0, origin)
            for node in path:
                scats_number = scats_site_data[scats_site_data['SCATS Number']. astype(str) == str(node)]
                if not scats_number.empty:
                    lat = scats_number.iloc[0]['NB_LATITUDE']
                    lon = scats_number.iloc[0]['NB_LONGITUDE']
                    path_coords.append((lat, lon))   
            colour = colours[idx % len(colours)]
            line = draw_route(map_widget, path_coords, colour)
            
            
            visible_var = tk.BooleanVar(value=True)
            
            path_data = {
                'path_coords': path_coords,
                'color': colour,
                'visible_var': visible_var,
                'line': line,
                'map_widget': map_widget
            }
            
            checkbox = tk.Checkbutton(toggle_frame, text=f"Path {idx+1}",
                                      variable=visible_var,
                                      command=partial(toggle_path_visibility, path_data))
            
            checkbox.pack(anchor='w')
            drawn_paths.append(path_data)
            
                
            idx+=1
    
    if pathtimes:
        travel_time_label.pack(anchor='w', pady=5)
        j = 0
        for time in pathtimes:
            time_label = tk.Label(left_frame, text=f"Path {j+1} = {time} mins", font=('Arial', 15))
            time_label.pack(anchor='w', pady=3)
            j+=1
            drawn_path_times.append(time_label)
        
                    
def toggle_path_visibility(path_data):
    
    if path_data['visible_var'].get():
        #redraw the line if toggled on
        new_line = path_data['map_widget'].set_path(path_data['path_coords'], color=path_data['color'])
        path_data['line'] = new_line
    else:
        #remove the line if toggled off
        path_data['line'].delete()
        
def on_reset(toggle_frame, origin_opt, dest_opt, time_opt, model_opt, error_text, travel_time_label):
    
    global drawn_paths
    global drawn_path_times
    
    travel_time_label.pack_forget()
    
    for path in drawn_paths:
        path['line'].delete()
        
    drawn_paths.clear()
    
    for pathtime in drawn_path_times:
        pathtime.pack_forget()
    
    for widget in toggle_frame.winfo_children():
        widget.destroy()
    
    origin_opt.set('')
    dest_opt.set('')
    time_opt.set('')
    model_opt.set('')
    
    if error_text.winfo_ismapped():
        error_text.pack_forget()
        

def main():
    window = tk.Tk()

    top_frame = tk.Frame(window)
    top_frame.pack(side="top", pady=40)

    left_frame = tk.Frame(window)
    left_frame.pack(side="left", anchor="n", padx=20, pady=15)

    right_frame = tk.Frame(window)
    right_frame.pack(side="right", anchor="n", padx=20, pady=15)

    window.geometry("1500x1000")
    window.title("Traffic-based Route Guidance Problem")
    title = tk.Label(top_frame, text="Plan Your Route", font=('Arial', 40))
    title.pack()
    
    scats_site_data = pd.read_csv('Resources/scats_lat_lon_data.csv')
    scats_sites = scats_site_data['SCATS Number'].astype(str)
    scats_sites = scats_sites.to_list()
    
    origin_text = tk.Label(left_frame, text="Please select your starting point.", font=('Arial', 20))
    origin_text.pack(anchor="w", pady=10)

    origin_opt = tk.StringVar(value='')
    origin_option_menu = tk.OptionMenu(left_frame, origin_opt, *scats_sites)
    origin_option_menu.pack(anchor="w", pady=5)

    dest_text = tk.Label(left_frame, text="Please select your destination.", font=('Arial', 20))
    dest_text.pack(anchor="w", pady=10)

    dest_opt = tk.StringVar(value='')
    dest_option_menu = tk.OptionMenu(left_frame,dest_opt, *scats_sites)
    dest_option_menu.pack(anchor="w", pady=5)
    
    time_text = tk.Label(left_frame, text="Please select a time.", font=('Arial', 20))
    time_text.pack(anchor="w", pady=10)
    
    time_opt = tk.StringVar(value='')
    time_options = ['01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00',
                    '23:00', '24:00']
    time_option_menu = tk.OptionMenu(left_frame, time_opt, *time_options)
    time_option_menu.pack(anchor="w", pady=5)
    
    model_text = tk.Label(left_frame, text="Please select a model.", font=('Arial', 20))
    model_text.pack(anchor="w", pady=10)
    
    model_opt = tk.StringVar(value='')
    model_opt.trace_add("write", lambda *args: on_model_select(None, model_opt))
    model_options = ['LSTM', 'GRU', 'RNN']
    model_option_menu = tk.OptionMenu(left_frame, model_opt, *model_options)
    model_option_menu.pack(anchor="w", pady=5)

    error_text = tk.Label(left_frame, text="All fields must be selected", font=('Arial', 20))
    button = tk.Button(left_frame, text="Find Route", command=lambda: on_find_route(left_frame, map_widget, origin_opt, dest_opt, time_opt, model_opt, scats_site_data, error_text, toggle_frame, travel_time_label)) #add command property, pass in the function to execute action
    button.pack(anchor="w", pady=30)
    
    map_widget = TkinterMapView(right_frame, width=900, height=600)
    map_widget.pack(fill="both", padx=30)
    map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")  # OpenStreetMap (default)
    map_center_x = scats_site_data['NB_LATITUDE'].mean()
    map_center_y = scats_site_data['NB_LONGITUDE'].mean()
    map_widget.set_position(map_center_x, map_center_y)
    map_widget.set_zoom(13)
    
    toggle_frame = tk.Frame(map_widget)
    toggle_frame.place(relx=0.85, rely=0.01)
    
    reset_button = tk.Button(left_frame, text="Reset", command=lambda: on_reset(toggle_frame, origin_opt, dest_opt, time_opt, model_opt, error_text, travel_time_label))
    reset_button.pack(anchor='w', pady=10)
    
    travel_time_label = tk.Label(left_frame, text="Expected travel time:", font=('Arial', 17))
    
    
    draw_markers(map_widget, scats_site_data)


    window.mainloop()
    
    

if __name__ == "__main__":
    main()