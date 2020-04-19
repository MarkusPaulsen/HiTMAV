from models.Model.SourceLoaderContext import SourceLoaderContext

slc: SourceLoaderContext = SourceLoaderContext()
for image in slc.get_source_strategy().get_image_store():
    print(image.get_image_data())