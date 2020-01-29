/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lattice/Lattice_accessor.h

Copyright (C) 2015

Author: Gianluca Filaci <g.filaci@ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
			   /*  END LEGAL */
#pragma once 

#ifdef GRID_SYCL

// wrapper to sycl accessor in order to overload operator[]
// and access data through underlying OpenCL C pointer
template <class T>
class DeviceAccessor {
   public:
  
  typedef cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> sycl_device_accessor;
  
  sycl_device_accessor sycl_da;
  accelerator_inline       T & operator[](size_t i)       { return *(sycl_da.get_pointer().get() + i); };
  accelerator_inline const T & operator[](size_t i) const { return *(sycl_da.get_pointer().get() + i); };

   DeviceAccessor(sycl_device_accessor &sycl_da_) : sycl_da(sycl_da_) {}
   DeviceAccessor(cl::sycl::buffer<T,1> &buffer)  : sycl_da(buffer) {}
};

template <class T>
class Accessor {
 public:

  typedef T type;
  typedef cl::sycl::buffer <T, 1> accelerator_buffer;
  typedef DeviceAccessor<T> device_accessor;
  typedef cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> host_accessor;
    
  accelerator_buffer *buffer_ptr;
  device_accessor    *da_ptr;
  host_accessor      *ha_ptr;
  
  void set_accessor(const T* _odata, const uint64_t _odata_size) {
    buffer_ptr = new accelerator_buffer(_odata, cl::sycl::range<1>(_odata_size), cl::sycl::property::buffer::use_host_ptr());  // could pass aligned_allocator
    da_ptr     = new device_accessor(*buffer_ptr);
  }
  void delete_accessor() {
    delete buffer_ptr;
    delete da_ptr;
    buffer_ptr = nullptr;
    da_ptr     = nullptr;
  }
  device_accessor get_device_accessor() const {
    return *da_ptr;
  }
  void create_host_accessor() {
    ha_ptr = new host_accessor(const_cast<accelerator_buffer*>(buffer_ptr)->template get_access<cl::sycl::access::mode::read_write>());
  }
  void delete_host_accessor() {
    delete ha_ptr;
    ha_ptr = nullptr;
  }

  accelerator_inline       T & access_host(size_t i)       { return ha_ptr->operator[](i); }
  accelerator_inline const T & access_host(size_t i) const { return ha_ptr->operator[](i); }
  
 Accessor() : buffer_ptr(nullptr), da_ptr(nullptr), ha_ptr(nullptr) {}
};

#else

template <class T>
class Accessor {
 public:

  typedef T type;
  typedef T* device_accessor;
  typedef T* host_accessor;

  device_accessor da;
  host_accessor   ha;
  
  void set_accessor(T* _odata, uint64_t _odata_size) {
    da = _odata;
    ha = _odata;
  }
  void delete_accessor() {
    da = nullptr;
    ha = nullptr;
  }
  device_accessor get_device_accessor() const { return da; }
  void create_host_accessor() {}
  void delete_host_accessor() {}

  accelerator_inline       T & access_host(size_t i)       { return ha[i]; }
  accelerator_inline const T & access_host(size_t i) const { return ha[i]; }
  
 Accessor() : da(nullptr), ha(nullptr) {}
};

#endif
