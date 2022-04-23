import tkinter
from tkinter import *
from tkinter.ttk import *
import Model

Main_Window = Tk()
Main_Window.call("source", "sun-valley.tcl")
Main_Window.call("set_theme", "light")
Main_Window.title('Backward Propagation')
Main_Window.geometry("900x400")


NumberOfHiddenLayer = StringVar()
NeuronsPerLayers = StringVar()
Epochs = StringVar()
LearnigRate = StringVar()
ActivationFunction = StringVar()
Bias  = StringVar()
HasBias = IntVar(value=0)
ActivationFunctionList = ['Sigmoid','Hyperbolic Tangent sigmoid']

def change_theme():
    if Main_Window.tk.call("ttk::style", "theme", "use") == "sun-valley-dark":
        Main_Window.tk.call("set_theme", "light")

    else:
        Main_Window.tk.call("set_theme", "dark")

def SetTrainData():
   Bp = Model.Model(NumberOfHiddenLayer.get(),NeuronsPerLayers.get() , Epochs.get() , LearnigRate.get(),ActivationFunction.get(),Bias.get(),HasBias.get())
   Bp.PreProcessing()
   Bp.CreateModel()
   Bp.Train()

def SetTestData():
   Bp = Model.Model(NumberOfHiddenLayer.get(),NeuronsPerLayers.get() , Epochs.get() , LearnigRate.get(),ActivationFunction.get(),Bias.get(),HasBias.get())
   Bp.PreProcessing()
   Bp.CreateModel()
   Bp.Test()

# Layers
LayerLabel = Label(Main_Window,text="Layers Count",font=('calibre',10, 'bold')).place(x=5,y=10)
NumberOfLayerEntry = Entry(Main_Window,textvariable = NumberOfHiddenLayer, font=('calibre',10,'normal')).place(x=100,y=0)


#Neurons
NeuronsLabel = Label(Main_Window,text="Neurons",font=('calibre',10, 'bold')).place(x=280,y=10)
NeuronsCountForAllLayer = Entry(Main_Window,textvariable = NeuronsPerLayers, font=('calibre',10,'normal')).place(x=350,y=0)


#Function
FunctionLabel =Label(Main_Window,text="Function",font=('calibre',10, 'bold')).place(x=525,y=10)
FunctionComboBox = Combobox(Main_Window,width=27,textvariable=ActivationFunction)
FunctionComboBox.place(x=600,y=0)
FunctionComboBox['values'] = ActivationFunctionList
FunctionComboBox.current(0)


#Learning rate
LRLabel = Label(Main_Window,text="Learning Rate",font=('calibre',10, 'bold')).place(x=5,y=210)
LREntry = Entry(Main_Window,textvariable = LearnigRate, font=('calibre',10,'normal')).place(x=100,y=200)


#Epochs
EpochsLabel = Label(Main_Window,text="Epochs",font=('calibre',10, 'bold')).place(x=280,y=210)
EpochsEntry = Entry(Main_Window,textvariable = Epochs, font=('calibre',10,'normal')).place(x=350,y=200)

#Bias
BiasEntry = Entry(Main_Window,textvariable = Bias, font=('calibre',10,'normal')).place(x=600,y=200)
BiasCheckBox = Checkbutton(Main_Window, text = "Bias",style="Switch.TCheckbutton",
                      variable = HasBias,
                      onvalue = 1,
                      offvalue = 0,
                      width = 5).place(x=525,y=200)


TrainButton = Button(Main_Window,text = 'Train', command = SetTrainData).place(x=350,y=300)


TestButton = Button(Main_Window,text = 'Test', command = SetTestData).place(x=250,y=300)

ChangeThemeButton = Button(Main_Window, text="Change theme", command=change_theme).place(x=450,y=300)

Main_Window.mainloop()