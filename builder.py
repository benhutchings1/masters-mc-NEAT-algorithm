from src.Fitness import generate_building
from src.Blocks import block_interactions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print(event.event_type, event.src_path)

    def on_created(self, event):
        print("on_created", event.src_path)
        print(event.src_path.strip())
        if((event.src_path).strip() == ".\test.xml"):        
            print("Execute your logic here!")

event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path='./checkpoint', recursive=False)
observer.start()


while True:
    try:
        pass
    except KeyboardInterrupt:
        observer.stop()