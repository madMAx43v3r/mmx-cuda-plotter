#ifndef INCLUDE_MMX_STDIOX_HPP_
#define INCLUDE_MMX_STDIOX_HPP_

#include <stdio.h>
#include <fcntl.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#define OPEN(...) _open(__VA_ARGS__)
#define CLOSE(...) _close(__VA_ARGS__)
#define READ(...) _read(__VA_ARGS__)
#define WRITE(...) _write(__VA_ARGS__)
#define LSEEK(...) _lseeki64(__VA_ARGS__)
#define FSEEK(...) _fseeki64(__VA_ARGS__)
#define FTELL(...) ::_ftelli64(__VA_ARGS__)
#define FSYNC(...) ::_commit(__VA_ARGS__)
#define CLOSESOCKET(...) closesocket(__VA_ARGS__)
#define O_DIRECT 0
#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#undef PSIZE
#else
#include <error.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#define OPEN(...) ::open(__VA_ARGS__)
#define CLOSE(...) ::close(__VA_ARGS__)
#define READ(...) ::read(__VA_ARGS__)
#define WRITE(...) ::write(__VA_ARGS__)
#define LSEEK(...) ::lseek(__VA_ARGS__)
#define FSEEK(...) ::fseek(__VA_ARGS__)
#define FTELL(...) ::ftell(__VA_ARGS__)
#define FSYNC(...) ::fsync(__VA_ARGS__)
#define CLOSESOCKET(...) ::close(__VA_ARGS__)
#define O_BINARY 0
#endif

#endif // INCLUDE_MMX_STDIOX_HPP_
