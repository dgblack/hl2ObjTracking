import os

str1 = 'xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"'
str2 = '<rescap:Capability Name="perceptionSensorsExperimental" />'
fpath = ["C:/fakeUser/fakePath/ProjectName/Build/ProjectName/Package.appxmanifest"]

for i in range(len(fpath)):
    if os.path.exists(fpath[i]):
        text = ""
        print("Changing " + fpath[i])
        with open(fpath[i],"r") as file:
            text = file.read()
            txts = text.split(" IgnorableNamespaces")
            text = txts[0] + " " + str1 + " IgnorableNamespaces" + txts[1]
            txts = text.split('<DeviceCapability Name="webcam" />')
            text = txts[0] + str2 + '\n    <DeviceCapability Name="webcam" />' + txts[1]
        with open(fpath[i],"w") as file:
            file.write(text)
