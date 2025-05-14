import pandas as pd
import tkinter as tk
from tkintermapview import TkinterMapView

def draw_markers(map_widget, scats_site_data):
   for _, row in scats_site_data.iterrows():
        map_widget.set_marker(
            row["NB_LATITUDE"],
            row["NB_LONGITUDE"],
            text=f"{row['SCATS Number']}"
        )
def draw_route(map_widget, path_coords):
    map_widget.set_path(path_coords)
    
    lats = [coord[0] for coord in path_coords]
    lons = [coord[1] for coord in path_coords]

    # Calculate bounding box
    top_left = (max(lats), min(lons))      # Most north, most west
    bottom_right = (min(lats), max(lons))  # Most south, most east

    # Fit map view to bounding box
    map_widget.fit_bounding_box(top_left, bottom_right)

def on_find_route(left_frame, map_widget, origin_opt, dest_opt, time_opt, model_opt, scats_site_data, error_text):
    origin = origin_opt.get()
    destination = dest_opt.get()
    time = time_opt.get()
    model = model_opt.get()
    
    if error_text.winfo_ismapped():
        error_text.pack_forget()
    
    if not origin or not destination or not time or not model:
        if not error_text.winfo_ismapped():  # Check if the label is already packed
            error_text.pack(anchor='w', pady=30)
        print("All fields must be selected")
        return
    
    selection_text = tk.Label(left_frame, text=f"Origin: {origin}, Destination: {destination}, Time: {time}, Model: {model}", font=('Arial', 20))
    selection_text.pack(anchor="w", pady=30)
    print(f"Origin: {origin}, Destination: {destination}, Time: {time}, Model: {model}")
    
    #placeholder path, supposed to be dynamically retrieved
    path = path = [
    (scats_site_data.iloc[0]["NB_LATITUDE"], scats_site_data.iloc[0]["NB_LONGITUDE"]),
    (scats_site_data.iloc[16]["NB_LATITUDE"], scats_site_data.iloc[16]["NB_LONGITUDE"]),
    (scats_site_data.iloc[1]["NB_LATITUDE"], scats_site_data.iloc[1]["NB_LONGITUDE"])]
    
    draw_route(map_widget, path)
    

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
    title = tk.Label(top_frame, text="Welcome", font=('Arial', 40))
    title.pack()
    
    scats_site_data = pd.read_csv('Resources/scats_lat_lon_data.csv')
    scats_sites = scats_site_data['SCATS Number'].astype(str)
    scats_sites = scats_sites.to_list()
    
    origin_text = tk.Label(left_frame, text="Please select your starting point.", font=('Arial', 20))
    origin_text.pack(anchor="w", pady=20)

    origin_opt = tk.StringVar(value='')
    origin_option_menu = tk.OptionMenu(left_frame, origin_opt, *scats_sites)
    origin_option_menu.pack(anchor="w", pady=5)

    dest_text = tk.Label(left_frame, text="Please select your destination.", font=('Arial', 20))
    dest_text.pack(anchor="w", pady=20)

    dest_opt = tk.StringVar(value='')
    dest_option_menu = tk.OptionMenu(left_frame,dest_opt, *scats_sites)
    dest_option_menu.pack(anchor="w", pady=5)
    
    time_text = tk.Label(left_frame, text="Please select a time.", font=('Arial', 20))
    time_text.pack(anchor="w", pady=20)
    
    time_opt = tk.StringVar(value='')
    time_options = ['01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00',
                    '23:00', '24:00']
    time_option_menu = tk.OptionMenu(left_frame, time_opt, *time_options)
    time_option_menu.pack(anchor="w", pady=5)
    
    model_text = tk.Label(left_frame, text="Please select a model.", font=('Arial', 20))
    model_text.pack(anchor="w", pady=20)
    
    model_opt = tk.StringVar(value='')
    model_options = ['LSTM', 'GRU', 'RNN']
    model_option_menu = tk.OptionMenu(left_frame, model_opt, *model_options)
    model_option_menu.pack(anchor="w", pady=5)
    
    error_text = tk.Label(left_frame, text="All fields must be selected", font=('Arial', 20))
    button = tk.Button(left_frame, text="Find Route", command=lambda: on_find_route(left_frame, map_widget, origin_opt, dest_opt, time_opt, model_opt, scats_site_data, error_text)) #add command property, pass in the function to execute action
    button.pack(anchor="w", pady=30)
    
    map_widget = TkinterMapView(right_frame, width=900, height=600)
    map_widget.pack(fill="both", padx=30)
    map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")  # OpenStreetMap (default)
    map_center_x = scats_site_data['NB_LATITUDE'].mean()
    map_center_y = scats_site_data['NB_LONGITUDE'].mean()
    map_widget.set_position(map_center_x, map_center_y)
    map_widget.set_zoom(13)
    
    
    draw_markers(map_widget, scats_site_data)
    # draw_route(map_widget, path)


    window.mainloop()
    
    

if __name__ == "__main__":
    main()