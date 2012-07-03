/** @file vtkCudaFunctionObject.h
 *
 *  @brief Header file defining the abstract interface for defining 2D transfer function objects
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note These objects hold the required information for defining a 2D transfer function with multiple colours, objects, etc...
 *
 */
#ifndef VTKCUDAFUNCTIONOBJECT_H
#define VTKCUDAFUNCTIONOBJECT_H

#include "vtkObject.h"

/** @brief Function objects are contiguous regions with equal colour, opacity and identifier in a 2D transfer function. They are used to define discernable structures in the ray caster and voxel classifier.
 *
 */
class vtkCudaFunctionObject : public vtkObject {
public:

	vtkTypeMacro( vtkCudaFunctionObject, vtkObject );
	
	/** @brief Gets the classifer value associated with this particular function object
	 *
	 *  @return A positive short
	 */
	short	GetIdentifier();
	
	/** @brief Sets the classifer value associated with this particular function object
	 *
	 *  @param id The desired classifier for this object
	 *
	 *  @pre \a id is positive (and is not zero)
	 *
	 *  @todo Error macro required for reporting invalid identifier values
	 */
	void	SetIdentifier(short id);
	
	/** @brief Gets the red colour value associated with this object
	 *
	 *  @return A floating point between 0.0f and 1.0f inclusive representing the red colour value of this object
	 */
	float	GetRedColourValue();

	/** @brief Gets the green colour value associated with this object
	 *
	 *  @return A floating point between 0.0f and 1.0f inclusive representing the green colour value of this object
	 */
	float	GetGreenColourValue();

	/** @brief Gets the blue colour value associated with this object
	 *
	 *  @return A floating point between 0.0f and 1.0f inclusive representing the blue colour value of this object
	 */
	float	GetBlueColourValue();

	/** @brief Sets the colour values associated with this particular function object
	 *
	 *  @param R The desired red colour value for this object
	 *  @param G The desired green colour value for this object
	 *  @param B The desired blue colour value for this object
	 *
	 *  @pre All R, G and B are between 0.0f and 1.0f inclusive
	 */
	void	SetColour(float R, float G, float B);

	/** @brief Gets the opacity associated with this object
	 *
	 *  @return A floating point between 0.0f and 1.0f inclusive representing the opacity of this object
	 */
	float	GetOpacity();

	/** @brief Sets the opacity associated with this object
	 *
	 *  @param alpha The desired opacity of this particular object
	 *
	 *  @pre alpha is between 0.0f and 1.0f inclusive
	 */
	void	SetOpacity(float alpha);

	//accessors and mutators for the material-specific shading parameters
	vtkSetClampMacro( Ambient, float, 0.0f, 1.0f );
	vtkGetMacro( Ambient, float );
	vtkSetClampMacro( Diffuse, float, 0.0f, 1.0f );
	vtkGetMacro( Diffuse, float );
	vtkSetClampMacro( Specular, float, 0.0f, 1.0f );
	vtkGetMacro( Specular, float );
	vtkSetClampMacro( SpecularPower, float, 0.0f, 1.0f );
	vtkGetMacro( SpecularPower, float );

	/** @brief Method that, given a table to house the transfer function, applies the attributes (RGBA) to the parts of the table that are within the object
	 *
	 *  @param IntensitySize The size of each transfer function lookup table in the intensity dimension
	 *  @param GradientSize The size of each transfer function lookup table in the gradient dimension
	 *  @param IntensityLow The minimum intensity represented by this table (often the minimum intensity of the image)
	 *  @param IntensityHigh The maximum intensity represented by this table (often the minimum maximum of the image)
	 *  @param GradientLow The minimum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientHigh The maximum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientOffset The offset for logarithmically scaling the gradient
	 *  @param rTable The transfer function lookup table housing the red colour component
	 *  @param gTable The transfer function lookup table housing the green colour component
	 *  @param bTable The transfer function lookup table housing the blue colour component
	 *  @param aTable The transfer function lookup table housing the opacity component
	 *
	 *  @pre rTable, gTable, bTable and aTable are all buffers of size IntensitySize*GradientSize
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual void	PopulatePortionOfTransferTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh, float IntensityOffset,
											float GradientLow, float GradientHigh, float GradientOffset,
											float* rTable, float* gTable, float* bTable, float* aTable,
											int logUsed ) = 0;
	
	/** @brief Method that, given a table to house the classification function, applies the identifer to the parts of the table that are within the object
	 *
	 *  @param IntensitySize The size of each classification function lookup table in the intensity dimension
	 *  @param GradientSize The size of each classification function lookup table in the gradient dimension
	 *  @param IntensityLow The minimum intensity represented by this table (often the minimum intensity of the image)
	 *  @param IntensityHigh The maximum intensity represented by this table (often the minimum maximum of the image)
	 *  @param GradientLow The minimum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientHigh The maximum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientOffset The offset for logarithmically scaling the gradient
	 *  @param table The classification function lookup table housing the label value
	 *
	 *  @pre tableis a buffer of size IntensitySize*GradientSize
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual void	PopulatePortionOfClassifyTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh, float IntensityOffset,
											float GradientLow, float GradientHigh, float GradientOffset,
											short* table, int logUsed) = 0;

	/** @brief Calculates largest gradient (not logarithmically scaled) which this particular object includes.
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual double	getMaxGradient() = 0;

	/** @brief Calculates smallest gradient (not logarithmically scaled) which this particular object includes.
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual double	getMinGradient() = 0;

	/** @brief Calculates largest intensity which this particular object includes.
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual double	getMaxIntensity() = 0;

	/** @brief Calculates smallest intensity which this particular object includes.
	 *
	 *  @note This function must be reimplemented for each subclass of the function object to account for changes in geometry
	 */
	virtual double	getMinIntensity() = 0;


protected:

	/** @brief Constructor which sets the main attributes of this object (colour and identifier) to 0
	 *
	 */
	vtkCudaFunctionObject();

	//general attributes for objects in a transfer/classification function
	short	identifier;		/**< The label given to voxels with intensity and gradient falling within this object */
	float	colourRed;		/**< The red component of the colour mapped to voxels with intensity and gradient falling within this object */
	float	colourGreen;	/**< The green component of the colour mapped to voxels with intensity and gradient falling within this object */
	float	colourBlue;		/**< The blue component of the colour mapped to voxels with intensity and gradient falling within this object */
	float	opacity;		/**< The opacity mapped to voxels with intensity and gradient falling within this object */
	
	//ADS shading paradigm parameters
	float	Ambient;		/**< The amount of ambient light reflected */
	float	Diffuse;		/**< The amount of diffuse light reflected */
	float	Specular;		/**< The amount of light scattered in a specular reflection */
	float	SpecularPower;	/**< The amount of scattering in the specular reflection (higher means less scatter) */
};


#endif