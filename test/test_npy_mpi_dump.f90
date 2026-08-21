! MPI smoke: each rank dumps owned SP slab; merge with merge_npy_shards.py
! Topology: MPI Cartesian, 0-based coords → filename (no legacy convert here).
! Equal local sizes. Marker: global 0-based I,J,K → v = 100*I + 10*J + K
!
! Example (adjust runner to your cluster):
!   fpm test --target test_npy_mpi_dump --profile release \
!     --runner mpirun --runner-args "-np 4" --flag "-L./lib"
!   python -m numpy_mpi.merge_npy_shards temperature _npy_mpi_smoke
!   python -m numpy_mpi.check_mpi_npy_smoke _npy_mpi_smoke 2 2 1 3 4 5
!
! Default process grid 2x2x1, local 3x4x5 → global 6x8x5 (np must be 4).

program test_npy_mpi_dump
   use, intrinsic :: iso_fortran_env, only: sp => real32, output_unit, error_unit
   use mpi_f08, only: MPI_Init, MPI_Comm_rank, MPI_Comm_size, MPI_COMM_WORLD, MPI_Cart_create, MPI_Cart_coords, MPI_Abort, &
         MPI_Barrier, MPI_Finalize, MPI_Comm, MPI_Comm_free
   use npy_dump_field, only: mpi_dump_field
   implicit none(type, external)

   integer, parameter :: ndims = 3
   integer, parameter :: npx = 2, npy = 2, npz = 1
   integer, parameter :: nx = 3, ny = 4, nz = 5
   character(len=*), parameter :: field_name = "temperature"
   character(len=*), parameter :: field_name_int = "cell_state"
   character(len=*), parameter :: outdir = "_npy_mpi_smoke"

   integer :: rank, nproc, ierr
   integer :: dims(ndims), coords(ndims)
   logical :: periods(ndims), reorder
   type(MPI_Comm) :: comm_cart
   real(kind=sp), allocatable :: field(:, :, :)
   integer, allocatable :: field_int(:, :, :)
   integer :: i, j, k, I0, J0, K0
   integer :: stat
   character(len=:), allocatable :: msg
   character(len=:), allocatable :: mkdir_cmd

   call MPI_Init(ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)

   if (nproc /= npx * npy * npz) then
      if (rank == 0) then
         write(error_unit, "(A,i0,A,i0)") &
               "FAIL need nproc=", npx * npy * npz, " got ", nproc
      end if
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
   end if

   dims = [npx, npy, npz]
   periods = [.false., .false., .false.]
   reorder = .false.
   call MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, comm_cart, ierr)
   call MPI_Cart_coords(comm_cart, rank, ndims, coords, ierr)

   ! owned only (no halo in this smoke)
   allocate(field(nx, ny, nz), source=0.0_sp)
   allocate(field_int(nx, ny, nz), source=0)

   I0 = coords(1) * nx
   J0 = coords(2) * ny
   K0 = coords(3) * nz
   do k = 1, nz
      do j = 1, ny
         do i = 1, nx
            field(i, j, k) = real(100 * (I0 + i - 1) + 10 * (J0 + j - 1) + (K0 + k - 1), sp)
            field_int(i, j, k) = 100 * (I0 + i - 1) + 10 * (J0 + j - 1) + (K0 + k - 1)
         end do
      end do
   end do

   if (rank == 0) then
      mkdir_cmd = "mkdir -p " // outdir
      call execute_command_line(mkdir_cmd, exitstat=stat)
      if (stat /= 0) then
         write(error_unit, "(A)") "FAIL mkdir " // outdir
         call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
   end if
   call MPI_Barrier(MPI_COMM_WORLD, ierr)

   call mpi_dump_field(field_name, coords, field, stat, msg, directory=outdir)
   if (stat /= 0) then
      write(error_unit, "(A,i0,A)") "FAIL rank ", rank, " dump: " // msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
   end if

   call mpi_dump_field(field_name_int, coords, field_int, stat, msg, directory=outdir)
   if (stat /= 0) then
      write(error_unit, "(A,i0,A)") "FAIL rank ", rank, " dump: " // msg
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
   end if

   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   if (rank == 0) then
      write(output_unit, "(A)") "OK fortran mpi npy dump smoke"
      write(output_unit, "(A,3i3)") "cart dims", npx, npy, npz
      write(output_unit, "(A,3i3)") "local nx ny nz", nx, ny, nz
      write(output_unit, "(A)") "dir " // outdir
      write(output_unit, "(A)") "merge: python -m numpy_mpi.merge_npy_shards temperature " // outdir
   end if

   call MPI_Comm_free(comm_cart, ierr)
   call MPI_Finalize(ierr)
end program test_npy_mpi_dump
