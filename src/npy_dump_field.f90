! Thin owned-field dump to topology-named .npy via stdlib.
! Caller passes 0-based (i,j,k) and the owned field (slice ghosts before call).

module npy_dump_field
   use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64
   !! Maps directly to precision.f90 in TESA
   use stdlib_io_npy, only: save_npy
   use npy_dump_names, only: make_npy_shard_name
   implicit none(type, external)
   private
   public :: mpi_dump_field

   interface mpi_dump_field
      !! Provides generic interface to dump a 3D field of type real, double precision, and integer, to npy
      module procedure npy_dump_field_sp
      module procedure npy_dump_field_dp
      module procedure npy_dump_field_int
   end interface mpi_dump_field

contains

   subroutine npy_dump_field_sp(field_name, rank_coords, field, stat, msg, directory)
      !! Save owned SP rank-3 field as {field_name}__i{i}_j{j}_k{k}.npy
      character(len=*), intent(in) :: field_name
      integer, intent(in) :: rank_coords(3)
      real(kind=sp), intent(in) :: field(:, :, :)
      integer, intent(out) :: stat
      character(len=:), allocatable, intent(out) :: msg
      character(len=*), intent(in), optional :: directory

      character(len=:), allocatable :: basename, path

      call make_npy_shard_name(field_name, rank_coords, basename, stat, msg)
      if (stat /= 0) return

      call join_dir(directory, basename, path)
      call save_npy(path, field, stat, msg)
      if (stat /= 0) then
         if (.not.allocated(msg)) msg = ""
         if (len_trim(msg) == 0) then
            msg = "npy_dump_field_sp: save_npy failed for " // path
         end if
      end if
   end subroutine npy_dump_field_sp

   subroutine npy_dump_field_dp(field_name, rank_coords, field, stat, msg, directory)
      !! Save owned DP rank-3 field as {field_name}__i{i}_j{j}_k{k}.npy
      character(len=*), intent(in) :: field_name
      integer, intent(in) :: rank_coords(3)
      real(kind=dp), intent(in) :: field(:, :, :)
      integer, intent(out) :: stat
      character(len=:), allocatable, intent(out) :: msg
      character(len=*), intent(in), optional :: directory

      character(len=:), allocatable :: basename, path

      call make_npy_shard_name(field_name, rank_coords, basename, stat, msg)
      if (stat /= 0) return

      call join_dir(directory, basename, path)
      call save_npy(path, field, stat, msg)
      if (stat /= 0) then
         if (.not.allocated(msg)) msg = ""
         if (len_trim(msg) == 0) then
            msg = "npy_dump_field_dp: save_npy failed for " // path
         end if
      end if
   end subroutine npy_dump_field_dp

   subroutine npy_dump_field_int(field_name, rank_coords, field, stat, msg, directory)
      !! Save owned integer rank-3 field as {field_name}__i{i}_j{j}_k{k}.npy
      character(len=*), intent(in) :: field_name
      integer, intent(in) :: rank_coords(3)
      integer, intent(in) :: field(:, :, :)
      integer, intent(out) :: stat
      character(len=:), allocatable, intent(out) :: msg
      character(len=*), intent(in), optional :: directory

      character(len=:), allocatable :: basename, path

      call make_npy_shard_name(field_name, rank_coords, basename, stat, msg)
      if (stat /= 0) return

      call join_dir(directory, basename, path)
      call save_npy(path, field, stat, msg)
      if (stat /= 0) then
         if (.not.allocated(msg)) msg = ""
         if (len_trim(msg) == 0) then
            msg = "npy_dump_field_int: save_npy failed for " // path
         end if
      end if
   end subroutine npy_dump_field_int

   subroutine join_dir(directory, basename, path)
      character(len=*), intent(in), optional :: directory
      character(len=*), intent(in) :: basename
      character(len=:), allocatable, intent(out) :: path
      integer :: n

      if (.not.present(directory)) then
         path = trim(basename)
         return
      end if
      n = len_trim(directory)
      if (n <= 0) then
         path = trim(basename)
         return
      end if
      if (directory(n:n) == "/" .or. directory(n:n) == "\") then
         path = directory(1:n) // trim(basename)
      else
         path = directory(1:n) // "/" // trim(basename)
      end if
   end subroutine join_dir

end module npy_dump_field
