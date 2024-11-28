import slicer



def load_extensions(extensions):
    manager = slicer.app.extensionsManagerModel()
    for extension in extensions:
        manager.downloadAndInstallExtensionByName(extension)


if __name__ == "__main__":
    extensions = [
        "DICOMwebBrowser",
        "MONAILabel",
        "DCMQI",
        "PETDICOMExtension",
        "SegmentEditorExtraEffects",
        #"SlicerDevelopmentToolbox",
        #"MarkupsToModel",
        #"Chest_Imaging_Platform",
        #"MONAIViz",
        #"MatlabBridge",
        #"NvidiaAIAssistedAnnotation",
        "PETTumorSegmentation",
        #"PyTorch",
        "QuantitativeReporting",
        #"SkullStripper",
        #"SlicerBatchAnonymize",
        #"SlicerRT",
        #"SlicerRadiomics",
        #"TorchIO",
        "TotalSegmentator",
        "MONAIAuto3DSeg",
        "SegmentWithSAM",
        "SlicerDcm2nii",
        "TCIABrowser",
        "nnUNet"
    ]

    load_extensions(extensions)