# Downloads and unpacks the four MNIST files into MNIST_DIR, skipping those already there:
#
#   cmake -DMNIST_DIR=<dir> -P cmake/DownloadMnist.cmake
#
# Run by the `mnist` target on every build (FLASHGRAD_DOWNLOAD_MNIST). A failed download only
# warns, so building offline still works; the examples that need MNIST skip it.
if(NOT MNIST_DIR)
  message(FATAL_ERROR "DownloadMnist: pass -DMNIST_DIR=<dir>")
endif()

# The original host (yann.lecun.com) no longer serves the files; these mirror them unchanged.
set(mirrors
  https://ossci-datasets.s3.amazonaws.com/mnist
  https://storage.googleapis.com/cvdf-datasets/mnist
)
# SHA-256 of the .gz archives.
set(files
  train-images-idx3-ubyte 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
  train-labels-idx1-ubyte 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
  t10k-images-idx3-ubyte  8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
  t10k-labels-idx1-ubyte  f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
)

# Downloads <name>.gz from the first mirror that serves the expected bytes; sets `ok`.
function(download_archive name expected_hash archive ok)
  set(${ok} FALSE PARENT_SCOPE)
  foreach(mirror IN LISTS mirrors)
    file(DOWNLOAD "${mirror}/${name}.gz" "${archive}"
      STATUS status TIMEOUT 300 INACTIVITY_TIMEOUT 30)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
      list(GET status 1 reason)
      message(STATUS "MNIST: ${mirror}/${name}.gz failed: ${reason}")
      continue()
    endif()
    file(SHA256 "${archive}" hash)
    if(NOT hash STREQUAL expected_hash)
      message(STATUS "MNIST: ${mirror}/${name}.gz has an unexpected checksum")
      continue()
    endif()
    set(${ok} TRUE PARENT_SCOPE)
    return()
  endforeach()
endfunction()

find_program(GZIP gzip)

while(files)
  list(POP_FRONT files name expected_hash)
  set(target "${MNIST_DIR}/${name}")
  if(EXISTS "${target}")
    continue()
  endif()
  if(NOT GZIP)
    message(WARNING "MNIST: gzip not found, cannot unpack; skipping the download")
    return()
  endif()

  message(STATUS "MNIST: downloading ${name}")
  file(MAKE_DIRECTORY "${MNIST_DIR}")
  set(archive "${target}.gz.part")
  download_archive(${name} ${expected_hash} "${archive}" ok)
  if(NOT ok)
    file(REMOVE "${archive}")
    message(WARNING "MNIST: could not download ${name}; the MNIST examples will skip "
                    "it. The next build tries again (-DFLASHGRAD_DOWNLOAD_MNIST=OFF "
                    "turns this off).")
    return()
  endif()

  # Unpack next to the target and rename, so an interrupted build never leaves a partial file
  # under the final name.
  execute_process(COMMAND "${GZIP}" -dc "${archive}"
    OUTPUT_FILE "${target}.part" RESULT_VARIABLE result)
  file(REMOVE "${archive}")
  if(NOT result EQUAL 0)
    file(REMOVE "${target}.part")
    message(WARNING "MNIST: could not unpack ${name}")
    return()
  endif()
  file(RENAME "${target}.part" "${target}")
endwhile()
