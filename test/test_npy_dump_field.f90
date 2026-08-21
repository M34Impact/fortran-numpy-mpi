! Serial smoke: npy_dump_field_sp with halo-owned slice + topology name.
! Python check (from repo root, after test writes cwd file):
!   python helpers/npy_names.py temperature__i1_j2_k0.npy
!   python helpers/check_npy_marker.py temperature__i1_j2_k0.npy
! (marker values differ from rung-1 formula if you only check shape/dtype —
!  this test uses the same 100*i+10*j+k local owned marker.)

program test_npy_dump_field
   use, intrinsic :: iso_fortran_env, only: sp => real32, output_unit
   use stdlib_io_npy, only: load_npy
   use npy_dump_field, only: mpi_dump_field
   implicit none

   integer, parameter :: nx = 3, ny = 4, nz = 5
   integer, parameter :: mpi_coords(3) = [1, 2, 0]
   character(len=*), parameter :: field = "temperature"

   real(sp), allocatable :: a(:, :, :), b(:, :, :)
   integer :: i, j, k, stat
   character(len=:), allocatable :: msg
   real(sp) :: expect
   logical :: ok
   integer :: nfail

   allocate (a(0:nx + 1, 0:ny + 1, 0:nz + 1), source=0.0_sp)

   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            a(i, j, k) = 100.0_sp*real(i, sp) + 10.0_sp*real(j, sp) + real(k, sp)
         end do
      end do
   end do

   call mpi_dump_field(field, mpi_coords, a(1:nx, 1:ny, 1:nz), stat, msg)
   if (stat /= 0) then
      write (output_unit, "(a)") "FAIL dump: " // msg
      stop 1
   end if

   call load_npy("temperature__i1_j2_k0.npy", b, stat, msg)
   if (stat /= 0) then
      write (output_unit, "(a)") "FAIL load_npy: " // msg
      stop 1
   end if

   ok = .true.
   if (size(b, 1) /= nx .or. size(b, 2) /= ny .or. size(b, 3) /= nz) then
      write (output_unit, "(a,3i6)") "FAIL shape want", nx, ny, nz
      write (output_unit, "(a,3i6)") "FAIL shape got ", shape(b)
      ok = .false.
   end if

   nfail = 0
   if (ok) then
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               expect = 100.0_sp*real(i, sp) + 10.0_sp*real(j, sp) + real(k, sp)
               if (b(i, j, k) /= expect) then
                  nfail = nfail + 1
                  ok = .false.
                  if (nfail <= 5) then
                     write (output_unit, "(a,3i4,a,es16.8,a,es16.8)") &
                        "FAIL idx", i, j, k, " got ", b(i, j, k), " want ", expect
                  end if
               end if
            end do
         end do
      end do
   end if

   if (.not. ok) then
      write (output_unit, "(a,i0)") "FAIL npy_dump_field round-trip nfail=", nfail
      stop 1
   end if

   write (output_unit, "(a)") "OK fortran npy_dump_field_sp"
   write (output_unit, "(a)") "wrote temperature__i1_j2_k0.npy"
   write (output_unit, "(a)") "loaded temperature__i1_j2_k0.npy"
   write (output_unit, "(a,3i4)") "topology i j k", mpi_coords
   write (output_unit, "(a,3i4)") "owned shape", nx, ny, nz
   write (output_unit, "(a,es16.8)") "sample b(1,1,1)=", b(1, 1, 1)
   write (output_unit, "(a,es16.8)") "sample b(nx,ny,nz)=", b(nx, ny, nz)
end program test_npy_dump_field
