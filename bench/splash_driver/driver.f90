program drv
 use projections3D, only:interpolate3D_projection
 implicit none
 integer :: n,npix,iu,t0,t1,rate,ir,nrep
 real :: xmin,ymin,pw,tsum
 real, allocatable :: x(:),y(:),z(:),h(:),w(:),d(:),img(:,:)
 integer, allocatable :: it(:)
 character(len=256) :: fin,fout,arg
 call get_command_argument(1,fin); call get_command_argument(2,fout); call get_command_argument(3,arg); read(arg,*) nrep
 open(newunit=iu,file=trim(fin),form='unformatted',access='stream',status='old')
 read(iu) n,npix,xmin,ymin,pw
 allocate(x(n),y(n),z(n),h(n),w(n),d(n),it(n),img(npix,npix))
 read(iu) x,y,z,h,w,d
 close(iu)
 it = 1
 call system_clock(t0,rate)
 do ir=1,nrep
    call interpolate3D_projection(x,y,z,h,w,d,it,n,xmin,ymin,img,npix,npix,pw,pw,.false.,1.e30,0.,.false.,.false.,-1)
    if (ir==1) then
       call system_clock(t1); write(*,'(a,f8.3)') 'first call s: ',real(t1-t0)/rate
    endif
 enddo
 call system_clock(t1)
 write(*,'(a,f8.3)') 'mean s per call: ',real(t1-t0)/rate/nrep
 open(newunit=iu,file=trim(fout),form='unformatted',access='stream',status='replace')
 write(iu) img
 close(iu)
end program
