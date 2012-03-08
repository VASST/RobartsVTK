
#ifndef GLSLSHADER_H
#define GLSLSHADER_H

#include <string>

class GLSLShader
{
  GLuint mProgramName;
  GLuint mVertexShaderName;
  GLuint mFragmentShaderName;

  std::string mErrorLog;

public:
  GLSLShader()
  {
    mProgramName = 0;
    mVertexShaderName = glCreateShader(GL_VERTEX_SHADER);
    mFragmentShaderName = glCreateShader(GL_FRAGMENT_SHADER);
    mErrorLog = std::string("");
  }

  GLSLShader(std::string VSsource, std::string FSsource)
  {
    mVertexShaderName = glCreateShader(GL_VERTEX_SHADER);
    mFragmentShaderName = glCreateShader(GL_FRAGMENT_SHADER);
    LoadVertexShader(VSsource);
    LoadFragmentShader(FSsource);
    LinkProgram();
  }

  std::string GetError(void)
  {
    return mErrorLog;
  }

  int LoadVertexShader(std::string source)
  {
    std::string code = std::string("");
    char line[512], *allsource;
    FILE *fp = fopen(source.c_str(),"r");

    // Load the code
    while(1)
    {
      fgets(line, 512, fp);
      if(feof(fp)) break;

      code = code + std::string(line);
    }
    fclose(fp);

    // Compile the code
    allsource = new char[strlen(code.c_str())+1];
    strcpy(allsource, code.c_str());
    glShaderSource(mVertexShaderName, 1, (const char **)&allsource, NULL);
    glCompileShader(mVertexShaderName);
    {
        GLint status;
        glGetShaderiv(mVertexShaderName, GL_COMPILE_STATUS, &status);
        GLint loglen;
        char log[1024];

        glGetShaderiv(mVertexShaderName, GL_INFO_LOG_LENGTH, &loglen);
        glGetShaderInfoLog(mVertexShaderName, 1024, &loglen, log);
        log[loglen] = '\0';
        mErrorLog = mErrorLog + std::string(log);

        if(status == GL_FALSE)
        {
            return 0;
        }
    }

    delete[] allsource;
    return 1;
  }

  int LoadFragmentShader(std::string source)
  {
    std::string code = std::string("");
    char line[512], *allsource;
    FILE *fp = fopen(source.c_str(),"r");

    // Load the code
    while(1)
    {
      fgets(line, 512, fp);
      if(feof(fp)) break;

      code = code + std::string(line);
    }
    fclose(fp);

    // Compile the code
    allsource = new char[strlen(code.c_str())+1];
    strcpy(allsource, code.c_str());
    glShaderSource(mFragmentShaderName, 1, (const char **)&allsource, NULL);
    glCompileShader(mFragmentShaderName);
    {
        GLint status;
        glGetShaderiv(mFragmentShaderName, GL_COMPILE_STATUS, &status);
        GLint loglen;
        char log[1024];

        glGetShaderiv(mFragmentShaderName, GL_INFO_LOG_LENGTH, &loglen);
        glGetShaderInfoLog(mFragmentShaderName, 1024, &loglen, log);
        log[loglen] = '\0';
        mErrorLog = mErrorLog + std::string(log);

        if(status == GL_FALSE)
        {
            return 0;
        }
    }

    delete[] allsource;
    return 1;
  }

  int LinkProgram(void)
  {
    mProgramName = glCreateProgram();
    glAttachShader(mProgramName, mVertexShaderName);
    glAttachShader(mProgramName, mFragmentShaderName);
    glLinkProgram(mProgramName);
    {
        GLint status;
        glGetProgramiv(mProgramName, GL_LINK_STATUS, &status);
        GLint loglen;
        char log[1024];

        glGetProgramiv(mProgramName, GL_INFO_LOG_LENGTH, &loglen);
        glGetProgramInfoLog(mProgramName, 1024, &loglen, log);
        log[loglen] = '\0';
        mErrorLog = mErrorLog + std::string(log);

        if(status == GL_FALSE)
        {
            return 0;
        }
    }

    return 1;
  }

  void Use(void)
  {
    glUseProgram(mProgramName);
  }

  static void Release(void)
  {
    glUseProgram(0);
  }

  GLuint GetProgram(void)
  {
    return mProgramName;
  }
};

#endif
