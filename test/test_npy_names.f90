! Smoke: make_npy_shard_name (module in src/npy_dump_names.f90)

program test_npy_names
   use, intrinsic :: iso_fortran_env, only: output_unit
   use npy_dump_names, only: make_npy_shard_name
   implicit none(type, external)

   character(len=:), allocatable :: name, msg
   integer :: mpi_coords(3)
   integer :: stat
   logical :: ok

   ok = .true.

   mpi_coords = [1, 2, 0]
   call make_npy_shard_name("temperature", mpi_coords, name, stat, msg)
   if (stat /= 0) then
      write(output_unit, "(a)") "FAIL " // msg
      stop 1
   end if
   if (name /= "temperature__i1_j2_k0.npy") then
      write(output_unit, "(a)") "FAIL got [" // name // "]"
      ok = .false.
   end if

   mpi_coords = [0, 0, 0]
   call make_npy_shard_name("c", mpi_coords, name, stat, msg)
   if (stat /= 0 .or. name /= "c__i0_j0_k0.npy") ok = .false.

   call make_npy_shard_name("", mpi_coords, name, stat, msg)
   if (stat == 0) ok = .false.

   call make_npy_shard_name("bad__i", mpi_coords, name, stat, msg)
   if (stat == 0) ok = .false.

   mpi_coords = [-1, 0, 0]
   call make_npy_shard_name("t", mpi_coords, name, stat, msg)
   if (stat == 0) ok = .false.

   if (.not.ok) then
      write(output_unit, "(a)") "FAIL fortran npy name builder"
      stop 1
   end if
   write(output_unit, "(a)") "OK fortran make_npy_shard_name"
   write(output_unit, "(a)") "sample temperature__i1_j2_k0.npy"
end program test_npy_names
