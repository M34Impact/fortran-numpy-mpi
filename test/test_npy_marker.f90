program test_npy_marker
   use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, output_unit
   use stdlib_io_npy, only: save_npy, load_npy
   implicit none(type, external)

   integer, parameter :: nx = 3, ny = 4, nz = 5
   character(len=*), parameter :: path = "marker_3d.npy"
   character(len=*), parameter :: path_sp = "marker_3d_sp.npy"

   real(kind=dp), allocatable :: a(:, :, :), b(:, :, :)
   real(kind=sp), allocatable :: a_sp(:, :, :), b_sp(:, :, :)
   integer :: i, j, k, stat, nfail
   character(len=:), allocatable :: msg
   real(kind=dp) :: expect
   real(kind=sp) :: espect
   logical :: ok

   allocate(a(0:nx + 1, 0:ny + 1, 0:nz + 1), source=0.0_dp)
   allocate(b(nx, ny, nz), source=0.0_dp)
   allocate(a_sp(0:nx + 1, 0:ny + 1, 0:nz + 1), source=0.0_sp)
   allocate(b_sp(nx, ny, nz), source=0.0_sp)

   ! Unique marker at each Fortran (i,j,k) — 1-based
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            a(i, j, k) = 100.0_dp * real(i, dp) + 10.0_dp * real(j, dp) + real(k, dp)
            a_sp(i, j, k) = 100.0_sp * real(i, sp) + 10.0_sp * real(j, sp) + real(k, sp)
         end do
      end do
   end do

   call save_npy(path, a(1:nx, 1:ny, 1:nz), stat, msg)
   if (stat /= 0) then
      write(output_unit, "(a)") "FAIL save_npy: " // msg
      stop 1
   end if

   call load_npy(path, b, stat, msg)
   if (stat /= 0) then
      write(output_unit, "(a)") "FAIL load_npy: " // msg
      stop 1
   end if

   call save_npy(path_sp, a_sp(1:nx, 1:ny, 1:nz), stat, msg)
   if (stat /= 0) then
      write(output_unit, "(a)") "FAIL save_npy: " // msg
      stop 1
   end if

   call load_npy(path_sp, b_sp, stat, msg)
   if (stat /= 0) then
      write(output_unit, "(a)") "FAIL load_npy: " // msg
      stop 1
   end if

   ok = .true.
   if (size(b, 1) /= nx .or. size(b, 2) /= ny .or. size(b, 3) /= nz) then
      write(output_unit, "(a,3i6)") "FAIL shape want", nx, ny, nz
      write(output_unit, "(a,3i6)") "FAIL shape got ", shape(b)
      ok = .false.
   end if

   nfail = 0
   if (ok) then
      do k = 1, nz
         do j = 1, ny
            do i = 1, nx
               expect = 100.0_dp * real(i, dp) + 10.0_dp * real(j, dp) + real(k, dp)
               if (b(i, j, k) /= expect) then
                  nfail = nfail + 1
                  if (nfail <= 5) then
                     write(output_unit, "(a,3i4,a,es24.16,a,es24.16)") &
                           "FAIL idx", i, j, k, " got ", b(i, j, k), " want ", expect
                  end if
                  ok = .false.
               end if

               espect = 100.0_sp * real(i, sp) + 10.0_sp * real(j, sp) + real(k, sp)
               if (b_sp(i, j, k) /= espect) then
                  nfail = nfail + 1
                  if (nfail <= 5) then
                     write(output_unit, "(a,3i4,a,es24.16,a,es24.16)") &
                           "FAIL idx", i, j, k, " got ", b_sp(i, j, k), " want ", espect
                  end if
                  ok = .false.
               end if
            end do
         end do
      end do
   end if

   if (.not.ok) then
      write(output_unit, "(a,i0)") "FAIL fortran round-trip, nfail=", nfail
      stop 1
   end if

   write(output_unit, "(a)") "OK fortran save_npy/load_npy marker round-trip"
   write(output_unit, "(a,3i4)") "shape", nx, ny, nz
   write(output_unit, "(a)") "wrote " // path
   write(output_unit, "(a,es24.16)") "sample a(1,1,1)=", a(1, 1, 1)
   write(output_unit, "(a,es24.16)") "sample a(nx,ny,nz)=", a(nx, ny, nz)
   write(output_unit, "(a)") "wrote " // path_sp
   write(output_unit, "(a,es24.16)") "sample a_sp(1,1,1)=", a_sp(1, 1, 1)
   write(output_unit, "(a,es24.16)") "sample a_sp(nx,ny,nz)=", a_sp(nx, ny, nz)
end program test_npy_marker
