#ifndef __VTKCUDAMEMORYTEXTURE_H__
#define __VTKCUDAMEMORYTEXTURE_H__

#include "vtkObject.h"
#include "vtkImageData.h"
#include "vector_types.h"

class vtkCudaMemoryTexture : public vtkObject
{
public:
    static vtkCudaMemoryTexture* New();

    void SetWidth(unsigned int width) { this->SetSize(width, this->GetHeight()); }
    void SetHeight(unsigned int height) { this->SetSize(this->GetWidth(), height); }
    void SetSize(unsigned int width, unsigned int height);

    unsigned int GetWidth() const { return this->Width; }
    unsigned int GetHeight() const { return this->Height; }

    unsigned int GetTexture() const { return this->TextureID; }

    void BindTexture();
    void BindBuffer();
    unsigned char* GetRenderDestination() const { return this->RenderDestination; }
    void UnbindBuffer();
    void UnbindTexture();

    bool CopyToVtkImageData(vtkImageData* data);

    typedef enum 
    {
        RenderToTexture,
        RenderToMemory,
    } RenderMode;
    void SetRenderMode(int mode);
    int GetCurrentRenderMode() const { return this->CurrentRenderMode; }


protected:
    vtkCudaMemoryTexture();
    ~vtkCudaMemoryTexture();

private:
    void Initialize();
    void RebuildBuffer();

private:
    unsigned char*	RenderDestination;

    unsigned int	TextureID;
    unsigned int	BufferObjectID;

    unsigned int	Width;
    unsigned int	Height;

    int				CurrentRenderMode;

    uchar4*			CudaOutputData;
    uchar4*			LocalOutputData;

    static bool  GLBufferObjectsAvailiable;
};
#endif /* __VTKCUDAMEMORYTEXTURE_H__ */
