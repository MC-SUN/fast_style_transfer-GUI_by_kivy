import kivy
from kivy.uix.widget import Widget
from kivy.app import App
from kivy.properties import ObjectProperty, NumericProperty, Clock
from kivy.uix.screenmanager import ScreenManager, Screen
import evaluate
from src import show_result
import transform_camera
style_camera_path="Assets/image/Snipaste157987527354359463image.jpg"
content_path="Assets/image/Snipaste157987527354359463image.jpg"
style_path="Assets/image/Snipaste157987527354359463image.jpg"
content_is_video=False
storage_location_path=""
not_busy=True
class Bar1(Widget):
	select_style_camera=ObjectProperty(None)
	select_style_image=ObjectProperty(None)
	status_camera=ObjectProperty(None)
	doing_camera=ObjectProperty(None)
	select_style_camera_stute=NumericProperty(0)
	def set_style_camera(self,p):#update function!
		if p!="Assets/image/Snipaste157987527354359463image.jpg":
			self.select_style_image.source=p
			self.select_style_camera_stute=1
			self.status_camera.text="Load image successfully.\nNow you can start cameraing\nyou can screen_shot by pressing \"S\",\nyou can back by pressing \"Q\",\nDefault storage dir: ../output/camera_screen/"
			#self.doing_camera.disabled = False
	def do_camera(self):
		self.status_camera.text = "Wait a few seconds, \n\nthe camera is loading..."
		camera_path = style_camera_path.split(".")[0] + ".ckpt"
		transform_camera.camera_transfer(camera_path)

class Bar2_left(Widget):
	is_content_ok=NumericProperty(0)
	is_style_ok = NumericProperty(0)
	content_image=ObjectProperty(None)
	style_image=ObjectProperty(None)
	select_content=ObjectProperty(None)
	select_style=ObjectProperty(None)
	status_content=ObjectProperty(None)
	status_style=ObjectProperty(None)
	def set_content(self,p):
		if p!="Assets/image/Snipaste157987527354359463image.jpg":
			if p.split(".")[1]=="mp4" or p.split(".")[1]=="avi":
				global content_is_video
				content_is_video=True
				self.content_image.source="Assets/image/Snipaste157987527354359463video.jpg"
			else:
				self.content_image.source=p
			self.is_content_ok=1
			self.status_content.text="Load content successfully."
	def set_style(self,p):
		if p != "Assets/image/Snipaste157987527354359463image.jpg":
			self.style_image.source=p
			self.is_style_ok=1
			self.status_style.text = "Load style successfully."
class Bar2_right(Widget):
	pass
class Bar3(Widget):
	is_storage_location_ok = NumericProperty(0)
	select_storage_location=ObjectProperty(None)
	status_location=ObjectProperty(None)
	doing_start=ObjectProperty(None)
	status_start=ObjectProperty(None)
	def set_location(self,p):
		if p!="":
			self.status_location.text=p
			self.is_storage_location_ok=1
	def do_start(self):
		style_checkpoint_path=style_path.split(".")[0]+".ckpt"
		out_path=storage_location_path+content_path.split('\\')[-1]
		if content_is_video:
			evaluate.ffwd_video(content_path,out_path,style_checkpoint_path)
			if show_result.show_video(out_path):
				return True
		else:
			evaluate.ffwd_to_img(content_path,out_path,style_checkpoint_path)
			if show_result.show_image(out_path):
				return True
		# time.sleep(10)
		'''
			界面变化有延迟！！！
			WAIT FOR WRITING!
			status=xxx(content_path,style_md_path,storage_location_path)
		'''
class Bar(Widget):
	bar_top=ObjectProperty(None)
	bar_middle = ObjectProperty(None)
	bar_bottom = ObjectProperty(None)

class Style_Camera_Selection(Screen):
	file_chooser = ObjectProperty(None)
	def get_camera_style(self):
		"""获取图片"""
		global style_camera_path
		style_camera_path=self.file_chooser.selection[0]
		self.manager.current = 'main_sc'
	def back(self):
		self.manager.current = 'main_sc'
class ContentSelection(Screen):
	file_chooser = ObjectProperty(None)
	def get_image_or_video(self):
		global content_path
		content_path = self.file_chooser.selection[0]
		self.manager.current = 'main_sc'
	def back(self):
		 self.manager.current = 'main_sc'
class StyleSelection(Screen):
	file_chooser = ObjectProperty(None)
	def get_image(self):
		"""获取图片"""
		global style_path
		style_path=self.file_chooser.selection[0]
		self.manager.current = 'main_sc'
	def back(self):
		self.manager.current = 'main_sc'
class LocationSelection(Screen):
	file_chooser = ObjectProperty(None)
	def get_location(self):
		"""获取存储位置"""
		global storage_location_path
		storage_location_path = self.file_chooser.path+'\\'
		self.manager.current = 'main_sc'
	def back(self):
		self.manager.current = 'main_sc'
class MainView(Screen):
	bar = ObjectProperty(None)
	def callback(self):
		self.bar.bar_top.select_style_camera.bind(on_press=self.select_image)
		self.bar.bar_top.doing_camera.bind(on_press=self.camera)
		self.bar.bar_middle.select_content.bind(on_press=self.content)
		self.bar.bar_middle.select_style.bind(on_press=self.style)
		self.bar.bar_bottom.select_storage_location.bind(on_press=self.storage)
		self.bar.bar_bottom.doing_start.bind(on_press=self.start)
	def select_image(self,mu):#mu:make up凑数用的
		"""Button:Select Style"""
		self.manager.current='sub1_sc'
	def camera(self,mu):
		"""Button:camera"""
		global not_busy
		not_busy=False
		self.bar.bar_top.do_camera()
		not_busy = True
	def disabled_switch(self,mu):
		global not_busy
		status=not not_busy
		self.bar.bar_top.select_style_camera.disabled = status
		self.bar.bar_middle.select_content.disabled = status
		self.bar.bar_middle.select_style.disabled = status
		self.bar.bar_bottom.select_storage_location.disabled =status
	def content(self,mu):
		"""Button:Content"""
		self.manager.current = 'sub2_sc'
	def style(self,mu):
		"""Button:Style"""
		self.manager.current = 'sub3_sc'
	def storage(self,mu):
		"""Button:Style"""
		self.manager.current = 'sub4_sc'
	def start(self,mu):
		global not_busy
		not_busy = False
		self.bar.bar_bottom.status_start.text = "Style transferring\nPlease wait..."
		if self.bar.bar_bottom.do_start():
			not_busy = True
			self.bar.bar_bottom.status_start.text = "Style transferring Done."
	def judge_status(self,mu):
		global not_busy
		self.disabled_switch(mu)
		if (self.bar.bar_middle.is_content_ok==1) and (self.bar.bar_middle.is_style_ok==1) and (self.bar.bar_bottom.is_storage_location_ok==1)and not_busy:
			self.bar.bar_bottom.doing_start.disabled=False
		if self.bar.bar_top.select_style_camera_stute==1 and not_busy:
			self.bar.bar_top.doing_camera.disabled=False
	def update(self,mu):
		self.bar.bar_top.set_style_camera(style_camera_path)
		self.bar.bar_middle.set_content(content_path)
		self.bar.bar_middle.set_style(style_path)
		self.bar.bar_bottom.set_location(storage_location_path)
		self.judge_status(mu)
class ISCApp(App):
	def build(self):
		sc=ScreenManager()
		sc1=MainView(name="main_sc")
		sc2=Style_Camera_Selection(name="sub1_sc")
		sc3=ContentSelection(name="sub2_sc")
		sc4=StyleSelection(name="sub3_sc")
		sc5=LocationSelection(name="sub4_sc")
		sc1.callback()
		Clock.schedule_interval(sc1.update,1.0/60)
		sc.add_widget(sc1)
		sc.add_widget(sc2)
		sc.add_widget(sc3)
		sc.add_widget(sc4)
		sc.add_widget(sc5)
		sc.current="main_sc"
		return sc
if __name__=='__main__':
	ISCApp().run()







#filechooser.FileChooserIconView