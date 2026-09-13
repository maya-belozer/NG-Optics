"""Render a repeatable GIF by exercising real Qt controls and mouse events."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QBuffer, QIODevice, QPoint, QPointF, Qt, QEvent
from PyQt6.QtGui import QFont, QFontDatabase, QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QScrollArea, QToolBar, QLineEdit
from optics2d.desktop import OpticsDesktop

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'media'
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont('Segoe UI', 10))
w = OpticsDesktop()
w.resize(1480, 980)
initial_uids = {item.uid for item in w.system.elements}
initial_horn_count = sum(item.kind == "horn" for item in w.system.elements)
w.show()
app.processEvents()
frames, durations = [], []
font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 24)
small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 16)

def capture(title, detail, duration=800, cursor=None):
 app.processEvents()
 buffer = QBuffer(); buffer.open(QIODevice.OpenModeFlag.WriteOnly)
 w.grab().save(buffer, 'PNG')
 shot = Image.open(io.BytesIO(bytes(buffer.data()))).convert('RGB')
 if cursor is not None:
  d = ImageDraw.Draw(shot);x,y=cursor.x(),cursor.y()
  d.ellipse((x-14,y-14,x+14,y+14),outline='#ef4444',width=3)
 canvas = Image.new('RGB',(shot.width,shot.height+84),'#0e1b2c')
 canvas.paste(shot,(0,84));d=ImageDraw.Draw(canvas)
 d.text((22,9),title,font=font,fill='#ffffff')
 d.text((22,46),detail,font=small,fill='#b9d3ed')
 canvas=canvas.resize((1184,round(canvas.height*1184/canvas.width)),Image.Resampling.LANCZOS)
 frames.append(canvas);durations.append(duration)

def button(text):
 return next(b for b in w.findChildren(QPushButton) if b.text()==text)

def click_button(b):
 scroll = w.findChild(QScrollArea)
 scroll.ensureWidgetVisible(b);app.processEvents()
 QTest.mouseClick(b,Qt.MouseButton.LeftButton);app.processEvents()

def add_horn():
 toolbar = next(t for t in w.findChildren(QToolBar) if t.windowTitle()=='Objects')
 action = next(a for a in toolbar.actions() if a.text()=='+ Horn')
 control=toolbar.widgetForAction(action)
 capture('1. Add a horn', 'Click + Horn in the toolbar to add a beam source.',1000,control.mapTo(w,control.rect().center()))
 QTest.mouseClick(control,Qt.MouseButton.LeftButton);app.processEvents()
 return w.selected()

capture('NG Optics v0.2.0', 'Start with the built-in Band 6 example and extend its optical layout.',1800)
horn=add_horn()
w.scene.setRange(xRange=(-160,400),yRange=(-180,180),padding=0)
app.processEvents()
capture('1. Add a horn', 'The new horn appears in the layout and its parameters are editable on the right.',1300)

def position(x,y):
 scene_point=w.scene.getViewBox().mapViewToScene(QPointF(x,y))
 return w.scene.mapFromScene(scene_point)

start=position(horn.x,horn.y)
end=position(horn.x-65,horn.y+60)
old=(horn.x,horn.y)
def event(kind,point,button,buttons):
 e=QMouseEvent(kind,QPointF(point),QPointF(w.scene.mapToGlobal(point)),button,buttons,Qt.KeyboardModifier.NoModifier)
 QApplication.sendEvent(w.scene.viewport(),e)
 app.processEvents()
event(QEvent.Type.MouseButtonPress,start,Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton)
for i in range(1,17):
 p=QPoint(round(start.x()+(end.x()-start.x())*i/16),round(start.y()+(end.y()-start.y())*i/16))
 event(QEvent.Type.MouseMove,p,Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton)
 capture('2. Drag the horn', 'Left-drag the source to reposition it; the beam updates with the layout.',70,w.scene.mapTo(w,p))
event(QEvent.Type.MouseButtonRelease,end,Qt.MouseButton.LeftButton,Qt.MouseButton.NoButton)
assert abs(horn.x-old[0])>30 and abs(horn.y-old[1])>30,(old,horn.x,horn.y)
capture('2. Drag the horn', 'The horn has moved. Use the coordinate fields for precise placement.',1200)
add=button('+ Frequency')
w.findChild(QScrollArea).ensureWidgetVisible(add);app.processEvents()
capture('3. Add a frequency channel', 'Scroll down in the horn properties and click + Frequency.',1500,add.mapTo(w,add.rect().center()))
click_button(add)
assert len(horn.frequency_channels)==2
w.findChild(QScrollArea).ensureWidgetVisible(w.frequency_table);app.processEvents()
capture('3. Add a frequency channel', 'Each horn supports multiple independently enabled frequency channels.',1200)

def edit_frequency(row,value):
 table=w.frequency_table
 w.findChild(QScrollArea).ensureWidgetVisible(table);app.processEvents()
 point=table.visualItemRect(table.item(row,1)).center()
 QTest.mouseClick(table.viewport(),Qt.MouseButton.LeftButton,pos=point)
 QTest.mouseDClick(table.viewport(),Qt.MouseButton.LeftButton,pos=point)
 app.processEvents()
 editor=table.findChild(QLineEdit)
 assert editor is not None
 QTest.keyClick(editor,Qt.Key.Key_A,Qt.KeyboardModifier.ControlModifier)
 QTest.keyClicks(editor,str(value))
 QTest.keyClick(editor,Qt.Key.Key_Return)
 app.processEvents()
 assert table.item(row,1).text()==str(value),table.item(row,1).text()

edit_frequency(0,230);edit_frequency(1,280)
capture('4. Set frequencies in GHz', 'Double-click the values: this horn will emit at 230 GHz and 280 GHz.',1700)
apply=button('Apply')
w.findChild(QScrollArea).ensureWidgetVisible(apply);app.processEvents()
capture('4. Apply the changes', 'Click Apply to update the Gaussian beam traces.',1000,apply.mapTo(w,apply.rect().center()))
click_button(apply)
assert [c.frequency_ghz for c in horn.frequency_channels]==[230,280]
w.findChild(QScrollArea).ensureWidgetVisible(w.frequency_table);app.processEvents()
capture('Two channels, one horn', '230 GHz and 280 GHz are now active. Add more horns to explore another beam path.',1900)
second=add_horn()
assert sum(e.kind == 'horn' for e in w.system.elements) == initial_horn_count + 2
assert initial_uids.issubset({e.uid for e in w.system.elements})
capture('Build up your optical layout', 'Add mirrors and lenses, connect the blocks, and save the system as JSON.',2200)
frames[0].save(OUT/'ng-optics-demo.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
frames[-3].save(OUT/'ng-optics-screenshot.png')
print(f'Saved {len(frames)} real UI frames; drag and frequency changes verified.')
w.close()
