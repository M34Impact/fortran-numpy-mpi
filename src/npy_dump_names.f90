! Filename builder for per-rank field dumps.
! Contract: {field}__i{i}_j{j}_k{k}.npy with 0-based topology coords.
! field: non-empty, [A-Za-z0-9_+=.-] only — no spaces (fail loud).
! No MPI. No I/O.

module npy_dump_names
   implicit none(type, external)
   private
   public :: make_npy_shard_name

   integer, parameter :: max_field_len = 128
   integer, parameter :: max_name_len = max_field_len + 64

contains

   pure subroutine make_npy_shard_name(field, rank_coords, filename, stat, msg)
      !! Build "{field}__i{i}_j{j}_k{k}.npy".
      !! On error: stat /= 0, msg set, filename left blank.
      character(len=*), intent(in) :: field
      integer, intent(in) :: rank_coords(3)
      character(len=:), allocatable, intent(out) :: filename
      integer, intent(out) :: stat
      character(len=:), allocatable, intent(out) :: msg

      character(len=max_name_len) :: buf
      character(len=32) :: si, sj, sk
      integer :: n, p
      character(len=1) :: ch

      stat = 0
      msg = ""
      filename = ""

      n = len_trim(field)
      if (n <= 0) then
         stat = 1
         msg = "make_npy_shard_name: empty field"
         return
      end if
      if (n > max_field_len) then
         stat = 2
         msg = "make_npy_shard_name: field too long"
         return
      end if
      if (index(field, "/") > 0 .or. index(field, "\") > 0) then
         stat = 3
         msg = "make_npy_shard_name: field must not contain path separators"
         return
      end if
      if (n >= 4) then
         if (field(n - 3:n) == ".npy") then
            stat = 4
            msg = "make_npy_shard_name: field must not include .npy"
            return
         end if
      end if
      if (index(field, "__i") > 0) then
         stat = 5
         msg = "make_npy_shard_name: field must not contain '__i'"
         return
      end if
      ! no whitespace / shell-unsafe (reject "cell state")
      do p = 1, n
         ch = field(p:p)
         if (ch == " " .or. ch == char(9) .or. ch == char(10) .or. ch == char(13)) then
            stat = 7
            msg = "make_npy_shard_name: field must not contain whitespace"
            return
         end if
         if (.not.( &
               (ch >= "A" .and. ch <= "Z") .or. &
               (ch >= "a" .and. ch <= "z") .or. &
               (ch >= "0" .and. ch <= "9") .or. &
               ch == "_" .or. ch == "+" .or. ch == "=" .or. &
               ch == "." .or. ch == "-")) then
            stat = 8
            msg = "make_npy_shard_name: field has unsafe character"
            return
         end if
      end do
      if (any(rank_coords < 0)) then
         stat = 6
         msg = "make_npy_shard_name: coords must be >= 0 (0-based topology)"
         return
      end if

      write(si, "(i0)") rank_coords(1)
      write(sj, "(i0)") rank_coords(2)
      write(sk, "(i0)") rank_coords(3)
      buf = trim(field) // "__i" // trim(si) // &
            "_j" // trim(sj) // &
            "_k" // trim(sk) // ".npy"
      filename = trim(buf)
   end subroutine make_npy_shard_name

end module npy_dump_names
