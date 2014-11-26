/** @file vtkCudaVolumeMapper.h
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#ifndef __vtkCudaVolumeMapper_H
#define __vtkCudaVolumeMapper_H

#include "vtkVolumeMapper.h"
#include "vtkCudaObject.h"
#include "vtkSetGet.h"

#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkCudaOutputImageInformationHandler.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_containerOutputImageInformation.h"

#include "vtkPlane.h"
#include "vtkPlanes.h"
#include "vtkPlaneCollection.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"

#include "vtkRenderer.h"
#include "vtkVolume.h"
#include <map>

// This is the maximum number of frames, may need to be set
#define VTKCUDAVOLUMEMAPPER_UPPER_BOUND 30

/** @brief vtkCudaVolumeMapper is a volume mapper, taking a set of 3D image data objects, volume and renderer as input and creates a 2D ray casted projection of the scene which is then displayed to screen
 *
 */
class vtkCudaVolumeMapper : public vtkVolumeMapper, public vtkCudaObject {
public:

	vtkTypeMacro( vtkCudaVolumeMapper, vtkVolumeMapper );

	/** @brief Sets the 3D image data for the first frame in the 4D sequence
	 *
	 *  @param image The 3D image data.
	 
	 *  @pre All dataset being rendered are the same size, anatomy, patient and modality
	 */
	virtual void SetInput( vtkImageData * image);

	/** @brief Sets the 3D image data for a particular frame in the 4D sequence
	 *
	 *  @param image The 3D image data.
	 *  @param frame The desired frame number when this data is rendered
	 *
	 *  @pre All dataset being rendered are the same size, anatomy, patient and modality
	 */
	void SetInput( vtkImageData * image, int frame);
	virtual void SetInputInternal( vtkImageData * image, int frame) = 0;

	/** @brief Sets the 3D image data for the first frame in the 4D sequence
	 
	 */
	virtual vtkImageData * GetInput();

	/** @brief Gets the 3D image data for a particular frame in the 4D sequence
	 *
	 */
	vtkImageData * GetInput( int frame);

	/** @brief Clears all the frames in the 4D sequence
	 *
	 */
	void ClearInput();
	virtual void ClearInputInternal() = 0;

	/** @brief Uses the provided renderer and volume to render the image data at the current frame
	 *
	 *  @note This is an internal method used primarily by the rendering pipeline
	 */
	void Render(vtkRenderer* renderer, vtkVolume* volume);
	
	/** @brief Perform specific rendering process
	 *
	 *  @note This is an internal method used primarily by the raycasting hierarchy structure
	 */
	virtual void InternalRender (	vtkRenderer* ren, vtkVolume* vol,
									const cudaRendererInformation& rendererInfo,
									const cudaVolumeInformation& volumeInfo,
									const cudaOutputImageInformation& outputInfo ) = 0;

	/** @brief Sets how the image is displayed which is passed to the output image information handler
	 *
	 *  @param scaleFactor The factor by which the screen is undersampled in each direction (must be equal or greater than 1.0f, where 1.0f means full sampling)
	 */
	void SetRenderOutputScaleFactor(float scaleFactor);
	
	/** @brief Set the strength and sensitivity parameters of the nonphotorealistic shading model which is given to the renderer information handler
	 *
	 *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
	 *  @param a The shading start value
	 *  @param b The shading stop value
	 */
	void SetCelShadingConstants(float darkness, float a, float b);

	/** @brief Set the strength and sensitivity parameters of the nonphotorealistic shading model which is given to the renderer information handler
	 *
	 *  @param darkness Floating point between 0.0f and 1.0f inclusive, where 0.0f means no shading, and 1.0f means maximal shading
	 *  @param a The shading start value
	 *  @param b The shading stop value
	 */
	void SetDistanceShadingConstants(float darkness, float a, float b);
	
	/** @brief Changes the next frame to be rendered to the provided frame
	 *
	 *  @param frame The next frame to be rendered
	 *
	 *  @pre frame is a non-negative integer less than the total number of frames
	 */
	void ChangeFrame(int frame);
	virtual void ChangeFrameInternal(int frame) = 0;

	/** @brief Gets the current frame being rendered
	 *
	 *  @post frame is a non-negative integer less than the total number of frames
	 */
	int GetCurrentFrame(){ return this->currFrame; };

	/** @brief Changes the next frame to be rendered to the next frame in the 4D sequence (modulo the number of frames, so if this is called on the last frame, the next frame is the first frame)
	 *
	 */
	void AdvanceFrame();

	/** @brief Changes the total number of frames being rendered (the number of 3D frames in the 4D sequence)
	 *
	 *  @param n The requested number of frames in the 4D sequence
	 *
	 *  @pre n is a non-negative integer less than 100
	 */
	void SetNumberOfFrames(int n);

	/** @brief Fetches the total number of frames being rendered (the number of 3D frames in the 4D sequence)
	 *
	 */
	int GetNumberOfFrames() {return this->numFrames;}
	
	/** @brief Gets a 2D image data consisting of the output of the most current render
	 *
	 *  @pre The volume mapper is currently rendering to vtkImageData (using the UseImageDataRenderering method), else this method returns NULL
	 */
	vtkImageData* GetOutput();

	// Description:
	// Specify Keyhole planes to be applied when the data is mapped
	// (at most 6 Keyhole planes can be specified).
	void AddKeyholePlane(vtkPlane *plane);
	void RemoveKeyholePlane(vtkPlane *plane);
	void RemoveAllKeyholePlanes();

	// Description:
	// Get/Set the vtkPlaneCollection which specifies the
	// Keyhole planes.
	virtual void SetKeyholePlanes(vtkPlaneCollection*);
	vtkGetObjectMacro(KeyholePlanes,vtkPlaneCollection);

	// Description:
	// An alternative way to set Keyhole planes: use up to six planes found
	// in the supplied instance of the implicit function vtkPlanes.
	void SetKeyholePlanes(vtkPlanes *planes);

	/** @brief Sets the displaying type to display to screen using CUDA-OpenGL interoperability, which is fast, but not always supported
	 *
	 *  @pre The rendering environment is ameniable to this form of rendering (aka, no multiple OpenGL contexts with isosurface object inclusion)
	 */
	void UseCUDAOpenGLInteroperability();

	/** @brief Sets the displaying type to display to screen using VTK's ray cast helper classes which is more generally supported
	 *
	 */
	void UseFullVTKCompatibility();

	/** @brief Sets the displaying type to not display to screen, but to save the image in a vtkImageData object which can be fetched using GetOutput()
	 *
	 *  @todo Support this option
	 */
	void UseImageDataRenderering();

	void SetImageFlipped(bool b){this->OutputInfoHandler->SetImageFlipped(b);};
	bool GetImageFlipped(){return this->OutputInfoHandler->GetImageFlipped();};

protected:
	/** @brief Constructor which initializes the number of frames, rendering type and other constants to safe initial values, and creates the required information handlers
	 *
	 */
	vtkCudaVolumeMapper();

	/** @brief Destructor which deallocates the various information handlers and matrices
	 *
	 */
	virtual ~vtkCudaVolumeMapper();

	virtual void Reinitialize(int withData = 0);
	virtual void Deinitialize(int withData = 0);

	vtkCudaRendererInformationHandler* RendererInfoHandler;		/**< The handler for any renderer/camera/geometry/clipping information */
	vtkCudaVolumeInformationHandler* VolumeInfoHandler;			/**< The handler for any volume/transfer function information */
	vtkCudaOutputImageInformationHandler* OutputInfoHandler;	/**< The handler for any output image housing/display information */

	vtkPlaneCollection*	KeyholePlanes;

	//modified time variables used to minimize setup
	unsigned long	renModified;								/**< The last time the renderer object was modified */
	unsigned long	volModified;								/**< The last time the volume object was modified */
	int	currFrame;									/**< The current frame being rendered */
	int	numFrames;									/**< The total number of frames housed by the mapper */

	/** @brief Using the mapper's volume and renderer objects, check for updates and reconstruct the appropriate matrices based on them, sending them off to the renderer information handler afterwards
	 *
	 *  @pre The mapper's volume and renderer objects are not null.
	 */
	void ComputeMatrices();
	vtkMatrix4x4	*ViewToVoxelsMatrix;						/**< Matrix used as temporary storage for the view to voxels transformation */
	vtkMatrix4x4	*WorldToVoxelsMatrix;						/**< Matrix used as temporary storage for the voxels to view transformation */

	vtkTransform	*PerspectiveTransform;						/**< Temporary storage of the perspective transform used by the current camera */
	vtkTransform	*VoxelsTransform;							/**< Temporary storage of the user defined volume transform used to modify postion, orientation, etc... */
	vtkTransform	*VoxelsToViewTransform;						/**< Temporary storage of the voxels to view transformation used to speed the process of switching/recalculating matrices*/
	vtkTransform	*NextVoxelsToViewTransform;					/**< Temporary storage of the next voxels to view transformation used to speed the process of switching/recalculating matrices */

	bool			erroredOut;									/**< Boolean to describe whether it is safe to render */
	bool			ImageFlipped;								/**< Boolean to describe whether the output image is flipped */

	std::map<int,vtkImageData*> inputImages;

private:

};

#endif
