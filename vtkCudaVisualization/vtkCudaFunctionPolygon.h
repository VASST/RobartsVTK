/** @file vtkCudaFunctionPolygon.h
*
 *  @brief Header file defining a polygon with fulfills the function object requirements
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */
#ifndef VTKCUDAFUNCTIONPOLYGON_H
#define VTKCUDAFUNCTIONPOLYGON_H

#include "vtkCudaFunctionObject.h"
#include <vector>
#include <list>

/** @brief Function object implemented as a polygon defined by an ordered set of vertices
 *
 *  @see vtkCudaFunctionObject
 *
 */
class vtkCudaFunctionPolygon : public vtkCudaFunctionObject {
public:

	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkCudaFunctionPolygon* New();

	/** @brief Method that, given a table to house the transfer function, applies the attributes (RGBA) to the parts of the table that are within the object reimplemented from vtkCudaFunctionObject
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
	 *  @see vtkCudaFunctionObject::PopulatePortionOfTransferTable()
	 *
	 */
	virtual void	PopulatePortionOfTransferTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh, float GradientOffset,
											float* rTable, float* gTable, float* bTable, float* aTable);

	/** @brief Method that, given a table to house the classification function, applies the identifer to the parts of the table that are within the object reimplemented from vtkCudaFunctionObject
	 *
	 *  @param IntensitySize The size of each transfer function lookup table in the intensity dimension
	 *  @param GradientSize The size of each transfer function lookup table in the gradient dimension
	 *  @param IntensityLow The minimum intensity represented by this table (often the minimum intensity of the image)
	 *  @param IntensityHigh The maximum intensity represented by this table (often the minimum maximum of the image)
	 *  @param GradientLow The minimum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientHigh The maximum logarithmically scaled gradient (including offset) represented by this table
	 *  @param GradientOffset The offset for logarithmically scaling the gradient
	 *  @param table The classification function lookup table housing the label value
	 *
	 *  @pre tableis a buffer of size IntensitySize*GradientSize
	 *
	 *  @see vtkCudaFunctionObject::PopulatePortionOfClassifyTable()
	 *
	 */
	virtual void	PopulatePortionOfClassifyTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh, float GradientOffset,
											short* table);

	/** @brief Calculates largest gradient (not logarithmically scaled) which this particular object includes.
	 *
	 *  @see vtkCudaFunctionObject::getMaxGradient()
	 */
	virtual double	getMaxGradient();

	/** @brief Calculates smallest gradient (not logarithmically scaled) which this particular object includes.
	 *
	 *  @see vtkCudaFunctionObject::getMinGradient()
	 */
	virtual double	getMinGradient();

	/** @brief Calculates largest intensity which this particular object includes.
	 *
	 *  @see vtkCudaFunctionObject::getMaxIntensity()
	 */
	virtual double	getMaxIntensity();

	/** @brief Calculates smallest intensity which this particular object includes.
	 *
	 *  @see vtkCudaFunctionObject::getMinIntensity()
	 */
	virtual double	getMinIntensity();
	
	/** @brief Returns the number of vertices currently in the polygon definition
	 *
	 */
	const int	GetNumVertices();
	
	/** @brief Adds a vertex to the polygon at a specific position (0 for front)
	 *
	 *  @param intensity The intensity value of this vertex
	 *  @param gradient The gradient value of this vertex
	 *  @param index The index of this added vertex after addition into the polygon
	 *
	 *  @pre index is a non-negative integer which is less than or equal to the number of vertices in the polygon before addition
	 *
	 *  @post The number of vertices increases by 1 after this method is called (and all preconditions are filled)
	 */
	void		AddVertex( float intensity, float gradient, unsigned int index );
	
	/** @brief Adds a vertex to the end position polygon
	 *
	 *  @param intensity The intensity value of this vertex
	 *  @param gradient The gradient value of this vertex
	 *
	 *  @post The number of vertices increases by 1 after this method is called
	 */
	void		AddVertex( float intensity, float gradient );

	/** @brief Modifies a vertex to the polygon at a specific position (0 for front)
	 *
	 *  @param intensity The intensity value of this vertex
	 *  @param gradient The gradient value of this vertex
	 *  @param index The index of this added vertex after addition into the polygon
	 *
	 *  @pre index is a non-negative integer which is less the number of vertices in the polygon
	 */
	void		ModifyVertex( float intensity, float gradient, unsigned int index );
	
	/** @brief Returns the intensity component of the vertex at the given index
	 *
	 *  @param index The index of this added vertex after addition into the polygon
	 *
	 *  @pre index is a non-negative integer which is less the number of vertices in the polygon
	 */
	const float	GetVertexIntensity( unsigned int index );

	/** @brief Returns the gradient component of the vertex at the given index
	 *
	 *  @param index The index of this added vertex after addition into the polygon
	 *
	 *  @pre index is a non-negative integer which is less the number of vertices in the polygon
	 */
	const float	GetVertexGradient( unsigned int index );
	
	/** @brief Removes the vertex at the given index from the polygon
	 *
	 *  @param index The index of this added vertex after addition into the polygon
	 *
	 *  @pre index is a non-negative integer which is less the number of vertices in the polygon
	 *  @pre The polygon must have one or more vertices
	 *
	 *  @post The number of vertices decreases by 1 after this method is called
	 */
	void		RemoveVertex( unsigned int index );

	
protected:
	
	/** @brief Struct containing information about a particular vertex
	 *
	 */
	struct vertex {
		float intensity;
		float gradient;
	};

	/** @brief Constructor which initializes the polygon to have no vertices
	 *
	 *  @see vtkCudaFunctionObject::vtkCudaFunctionObject
	 */
	vtkCudaFunctionPolygon();
	
	/** @brief Deconstructor which cleans up the additional memory used to store the vertices
	 *
	 */
	~vtkCudaFunctionPolygon();

	std::vector<vertex*> contour; /**< Container for the polygon's vertices */

	/** @brief Helper methods for determining whether a point is within the given polygon
	 *
	 */
	inline const bool pointInPolygon(const float x, const float y);



};
#endif