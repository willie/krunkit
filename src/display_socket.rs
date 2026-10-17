// SPDX-License-Identifier: Apache-2.0

//! A display and input channel to another process (Spock), for
//! `--display-socket <path>`. krunkit connects to the socket at <path>.
//!
//! Frames: the guest's scanout 0 is written into one of three buffers in the
//! file <path>.frames, which the other side maps. krunkit sends 32-byte
//! records of eight little-endian u32s:
//!   [1, width, height, format, stride, buffer_count, 0, 0]   configure
//!   [2, buffer, x, y, width, height, 0, 0]                    present
//!   [3, 0, 0, 0, 0, 0, 0, 0]                                  disable
//!   [4, width, height, hot_x, hot_y, 0, 0, 0]                 cursor
//!   [5, resource, width, height, format, stride, offset, size] blob scanout
//!   [6, resource, x, y, width, height, 0, 0]                  present from blob
//!   [7, resource, 0, 0, 0, 0, 0, 0]                           blob released
//! followed, for a cursor with a width, by its width x height BGRA pixels
//! (a width of 0 hides it). Formats are libkrun's (KRUN_DISPLAY_FORMAT_*,
//! the virtio-gpu ones). A blob scanout record carries the blob's descriptor
//! as SCM_RIGHTS ancillary data, once per resource; presents from it name
//! the resource instead of a frame buffer, and the frames file goes unused
//! while the guest scans out blobs. The receiver must read records with
//! recvmsg to get the descriptor.
//!
//! Input: the other side sends 12-byte records, little-endian:
//!   u16 device (0 keyboard, 1 pointer), u16 type, u16 code, u16 0, u32 value
//! with Linux evdev types and codes, including EV_SYN reports. Device 2 is
//! the display: a record there asks for the size `value >> 16` x
//! `value & 0xffff`, which the guest sees as a monitor change. The keyboard
//! reports every key; the pointer is a tablet with ABS_X/ABS_Y in
//! 0..=POINTER_MAX, five buttons, both wheels (with high-resolution
//! events, 120 a notch) and REL_X/REL_Y, so a guest that locks the pointer
//! gets relative motion.

use std::collections::{HashSet, VecDeque};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};

pub const POINTER_MAX: u32 = 32767;

const BUFFER_COUNT: usize = 3;

// Linux input-event-codes.h
const EV_KEY: u8 = 0x01;
const EV_REL: u8 = 0x02;
const EV_ABS: u8 = 0x03;
const KEY_MAX: usize = 0x2ff;
const BTN_LEFT: usize = 0x110;
const BTN_RIGHT: usize = 0x111;
const BTN_MIDDLE: usize = 0x112;
const BTN_SIDE: usize = 0x113;
const BTN_EXTRA: usize = 0x114;
const REL_X: usize = 0x00;
const REL_Y: usize = 0x01;
const REL_HWHEEL: usize = 0x06;
const REL_WHEEL: usize = 0x08;
const REL_WHEEL_HI_RES: usize = 0x0b;
const REL_HWHEEL_HI_RES: usize = 0x0c;
const ABS_X: u8 = 0x00;
const ABS_Y: u8 = 0x01;

#[link(name = "krun")]
extern "C" {
    fn krun_add_display(ctx_id: u32, width: u32, height: u32) -> i32;
    fn krun_set_display_backend(ctx_id: u32, backend: *const c_void, size: usize) -> i32;
    fn krun_display_set_dpi(ctx_id: u32, display_id: u32, dpi: u32) -> i32;
    fn krun_display_set_refresh_rate(ctx_id: u32, display_id: u32, refresh_rate: u32) -> i32;
    fn krun_add_input_device(
        ctx_id: u32,
        config: *const c_void,
        config_size: usize,
        events: *const c_void,
        events_size: usize,
    ) -> i32;
}

#[repr(C)]
struct KrunRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
struct DisplayBackend {
    features: u64,
    create_userdata: *const c_void,
    create: *const c_void,
    destroy: *const c_void,
    disable_scanout: extern "C" fn(*mut c_void, u32) -> i32,
    configure_scanout: extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32) -> i32,
    alloc_frame: extern "C" fn(*mut c_void, u32, *mut *mut u8, *mut usize) -> i32,
    present_frame: extern "C" fn(*mut c_void, u32, u32, *const KrunRect) -> i32,
    update_cursor: extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, *const u8) -> i32,
    get_resize_efd: extern "C" fn(*mut c_void) -> i32,
    take_display_size: extern "C" fn(*mut c_void, u32, *mut u32, *mut u32) -> i32,
    present_blob: extern "C" fn(*mut c_void, u32, u32, i32, u64, u32, u32, u32, u32, u64, *const KrunRect) -> i32,
    release_blob: extern "C" fn(*mut c_void, u32) -> i32,
}

#[repr(C)]
struct InputEvent {
    type_: u16,
    code: u16,
    value: u32,
}

#[repr(C)]
struct InputDeviceIds {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
}

#[repr(C)]
struct InputAbsInfo {
    min: u32,
    max: u32,
    fuzz: u32,
    flat: u32,
    res: u32,
}

type CreateFn = extern "C" fn(*mut *mut c_void, *const c_void, *const c_void) -> i32;

#[repr(C)]
struct InputConfig {
    features: u64,
    create_userdata: *const c_void,
    create: CreateFn,
    destroy: *const c_void,
    query_device_name: extern "C" fn(*mut c_void, *mut u8, usize) -> i32,
    query_serial_name: extern "C" fn(*mut c_void, *mut u8, usize) -> i32,
    query_device_ids: extern "C" fn(*mut c_void, *mut InputDeviceIds) -> i32,
    query_event_capabilities: extern "C" fn(*mut c_void, u8, *mut u8, usize) -> i32,
    query_abs_info: extern "C" fn(*mut c_void, u8, *mut InputAbsInfo) -> i32,
    query_properties: extern "C" fn(*mut c_void, *mut u8, usize) -> i32,
}

#[repr(C)]
struct InputEventProvider {
    features: u64,
    create_userdata: *const c_void,
    create: CreateFn,
    destroy: *const c_void,
    get_ready_efd: extern "C" fn(*mut c_void) -> i32,
    next_event: extern "C" fn(*mut c_void, *mut InputEvent) -> i32,
}

unsafe impl Sync for DisplayBackend {}
unsafe impl Sync for InputConfig {}
unsafe impl Sync for InputEventProvider {}

struct Frames {
    path: PathBuf,
    map: *mut u8,
    buffer_size: usize,
    next: usize,
    // (width, height, format) of the mapped buffers.
    shape: (u32, u32, u32),
}

unsafe impl Send for Frames {}

struct Channel {
    socket: Mutex<UnixStream>,
    frames: Mutex<Option<Frames>>,
    // Per input device: queued events and the pipe that is readable while
    // events are queued (one byte per event).
    queues: [Mutex<VecDeque<InputEvent>>; 2],
    pipes: [(File, File); 2],
    // A display size the other side asked for, and the pipe that is readable
    // while one is pending.
    pending_size: Mutex<Option<(u32, u32)>>,
    resize_pipe: (File, File),
    // Blobs whose descriptor the other side already has.
    blobs: Mutex<HashSet<u32>>,
}

static CHANNEL: OnceLock<Channel> = OnceLock::new();

fn channel() -> &'static Channel {
    CHANNEL.get().expect("display socket not set up")
}

fn send(record: [u32; 8]) {
    let mut bytes = [0u8; 32];
    for (i, v) in record.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    // The other side is gone: the VM keeps running without a display.
    let _ = channel().socket.lock().unwrap().write_all(&bytes);
}

/// Sends a record with a descriptor attached (SCM_RIGHTS).
fn send_with_fd(record: [u32; 8], fd: i32) -> bool {
    let mut bytes = [0u8; 32];
    for (i, v) in record.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let socket = channel().socket.lock().unwrap();
    let mut iov = libc::iovec { iov_base: bytes.as_mut_ptr() as *mut c_void, iov_len: bytes.len() };
    let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) } as usize;
    let mut control = vec![0u8; space];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr() as *mut c_void;
    msg.msg_controllen = space as _;
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as _;
        std::ptr::write_unaligned(libc::CMSG_DATA(cmsg) as *mut i32, fd);
        libc::sendmsg(socket.as_raw_fd(), &msg, 0) == bytes.len() as isize
    }
}

#[allow(clippy::too_many_arguments)]
extern "C" fn present_blob(
    _: *mut c_void,
    _scanout: u32,
    resource: u32,
    fd: i32,
    offset: u64,
    stride: u32,
    width: u32,
    height: u32,
    format: u32,
    size: u64,
    damage: *const KrunRect,
) -> i32 {
    let (Ok(offset), Ok(size)) = (u32::try_from(offset), u32::try_from(size)) else {
        return -4;
    };
    if channel().blobs.lock().unwrap().insert(resource) {
        if !send_with_fd([5, resource, width, height, format, stride, offset, size], fd) {
            channel().blobs.lock().unwrap().remove(&resource);
            return -1;
        }
    }
    let (x, y, w, h) = match unsafe { damage.as_ref() } {
        Some(r) => (r.x, r.y, r.width, r.height),
        None => (0, 0, u32::MAX, u32::MAX),
    };
    send([6, resource, x, y, w, h, 0, 0]);
    0
}

extern "C" fn release_blob(_: *mut c_void, resource: u32) -> i32 {
    if channel().blobs.lock().unwrap().remove(&resource) {
        send([7, resource, 0, 0, 0, 0, 0, 0]);
    }
    0
}

extern "C" fn disable_scanout(_: *mut c_void, _scanout: u32) -> i32 {
    send([3, 0, 0, 0, 0, 0, 0, 0]);
    0
}

extern "C" fn configure_scanout(
    _: *mut c_void,
    _scanout: u32,
    _display_width: u32,
    _display_height: u32,
    width: u32,
    height: u32,
    format: u32,
) -> i32 {
    let stride = width * 4;
    let buffer_size = (stride * height) as usize;
    let mut frames = channel().frames.lock().unwrap();
    // The guest sets the scanout on every page flip; only a new size or
    // format needs new buffers.
    if frames.as_ref().is_some_and(|f| !f.map.is_null() && f.shape == (width, height, format)) {
        return 0;
    }
    let path = frames.as_ref().map(|f| f.path.clone()).unwrap();
    if let Some(old) = frames.take() {
        unsafe { libc::munmap(old.map as *mut c_void, old.buffer_size * BUFFER_COUNT) };
    }
    // A new file, not the old one resized: the other side may still be
    // reading its mapping of the old one, which must not shrink under it.
    let _ = std::fs::remove_file(&path);
    let file = match File::options().read(true).write(true).create(true).truncate(true).open(&path) {
        Ok(f) => f,
        Err(e) => {
            log::error!("display socket: {}: {e}", path.display());
            return -1;
        }
    };
    if file.set_len((buffer_size * BUFFER_COUNT) as u64).is_err() {
        return -1;
    }
    let map = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            buffer_size * BUFFER_COUNT,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if map == libc::MAP_FAILED {
        return -1;
    }
    *frames = Some(Frames { path, map: map as *mut u8, buffer_size, next: 0, shape: (width, height, format) });
    drop(frames);
    send([1, width, height, format, stride, BUFFER_COUNT as u32, 0, 0]);
    0
}

extern "C" fn alloc_frame(_: *mut c_void, _scanout: u32, buffer: *mut *mut u8, size: *mut usize) -> i32 {
    let mut frames = channel().frames.lock().unwrap();
    let Some(f) = frames.as_mut() else { return -1 };
    let index = f.next;
    f.next = (f.next + 1) % BUFFER_COUNT;
    unsafe {
        *buffer = f.map.add(index * f.buffer_size);
        *size = f.buffer_size;
    }
    index as i32
}

extern "C" fn present_frame(_: *mut c_void, _scanout: u32, frame: u32, damage: *const KrunRect) -> i32 {
    let (x, y, w, h) = match unsafe { damage.as_ref() } {
        Some(r) => (r.x, r.y, r.width, r.height),
        None => (0, 0, u32::MAX, u32::MAX),
    };
    send([2, frame, x, y, w, h, 0, 0]);
    0
}

extern "C" fn update_cursor(
    _: *mut c_void,
    _scanout: u32,
    hot_x: u32,
    hot_y: u32,
    width: u32,
    height: u32,
    pixels: *const u8,
) -> i32 {
    let mut bytes = Vec::with_capacity(32 + (width * height * 4) as usize);
    for v in [4, width, height, hot_x, hot_y, 0, 0, 0] {
        bytes.extend_from_slice(&u32::to_le_bytes(v));
    }
    if width > 0 && !pixels.is_null() {
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(pixels, (width * height * 4) as usize) });
    }
    let _ = channel().socket.lock().unwrap().write_all(&bytes);
    0
}

extern "C" fn resize_fd(_: *mut c_void) -> i32 {
    channel().resize_pipe.0.as_raw_fd()
}

extern "C" fn take_display_size(_: *mut c_void, display: u32, width: *mut u32, height: *mut u32) -> i32 {
    if display != 0 {
        return 0;
    }
    let Some((w, h)) = channel().pending_size.lock().unwrap().take() else { return 0 };
    // Drain the wake-ups for this and any earlier request.
    let mut bytes = [0u8; 64];
    let fd = channel().resize_pipe.0.as_raw_fd();
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        while libc::read(fd, bytes.as_mut_ptr() as *mut c_void, bytes.len()) > 0 {}
    }
    unsafe {
        *width = w;
        *height = h;
    }
    1
}

static DISPLAY: DisplayBackend = DisplayBackend {
    // KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER | _CURSOR | _RESIZE | _BLOB_SCANOUT
    features: 1 | 2 | 4 | 8,
    create_userdata: std::ptr::null(),
    create: std::ptr::null(),
    destroy: std::ptr::null(),
    disable_scanout,
    configure_scanout,
    alloc_frame,
    present_frame,
    update_cursor,
    get_resize_efd: resize_fd,
    take_display_size,
    present_blob,
    release_blob,
};

// Input devices; the instance pointer is the device index.

extern "C" fn create(instance: *mut *mut c_void, userdata: *const c_void, _: *const c_void) -> i32 {
    unsafe { *instance = userdata as *mut c_void };
    0
}

fn device(instance: *mut c_void) -> usize {
    instance as usize
}

fn write_name(name: &str, buf: *mut u8, len: usize) -> i32 {
    let n = name.len().min(len);
    unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), buf, n) };
    n as i32
}

extern "C" fn device_name(instance: *mut c_void, buf: *mut u8, len: usize) -> i32 {
    write_name(["Spock keyboard", "Spock pointer"][device(instance)], buf, len)
}

extern "C" fn serial_name(instance: *mut c_void, buf: *mut u8, len: usize) -> i32 {
    write_name(["spock-kbd", "spock-ptr"][device(instance)], buf, len)
}

extern "C" fn device_ids(instance: *mut c_void, ids: *mut InputDeviceIds) -> i32 {
    unsafe {
        *ids = InputDeviceIds { bustype: 0x06, vendor: 0, product: device(instance) as u16 + 1, version: 1 };
    }
    0
}

fn set_bits(bits: &[usize], buf: *mut u8, len: usize) -> i32 {
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, len) };
    buf.fill(0);
    let mut used = 0;
    for &b in bits {
        if b / 8 < len {
            buf[b / 8] |= 1 << (b % 8);
            used = used.max(b / 8 + 1);
        }
    }
    used as i32
}

extern "C" fn event_capabilities(instance: *mut c_void, event_type: u8, buf: *mut u8, len: usize) -> i32 {
    match (device(instance), event_type) {
        (0, EV_KEY) => set_bits(&(1..BTN_LEFT).chain(BTN_EXTRA + 1..=KEY_MAX).collect::<Vec<_>>(), buf, len),
        (1, EV_KEY) => set_bits(&[BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_SIDE, BTN_EXTRA], buf, len),
        (1, EV_REL) => set_bits(&[REL_X, REL_Y, REL_HWHEEL, REL_WHEEL, REL_WHEEL_HI_RES, REL_HWHEEL_HI_RES], buf, len),
        (1, EV_ABS) => set_bits(&[ABS_X as usize, ABS_Y as usize], buf, len),
        _ => 0,
    }
}

extern "C" fn abs_info(_: *mut c_void, axis: u8, info: *mut InputAbsInfo) -> i32 {
    if axis != ABS_X && axis != ABS_Y {
        return -4; // KRUN_INPUT_ERR_INVALID_PARAM
    }
    unsafe { *info = InputAbsInfo { min: 0, max: POINTER_MAX, fuzz: 0, flat: 0, res: 0 } };
    0
}

extern "C" fn properties(_: *mut c_void, buf: *mut u8, len: usize) -> i32 {
    set_bits(&[], buf, len)
}

extern "C" fn ready_fd(instance: *mut c_void) -> i32 {
    channel().pipes[device(instance)].0.as_raw_fd()
}

extern "C" fn next_event(instance: *mut c_void, out: *mut InputEvent) -> i32 {
    let d = device(instance);
    let Some(event) = channel().queues[d].lock().unwrap().pop_front() else { return 0 };
    let mut byte = [0u8];
    let _ = (&channel().pipes[d].0).read(&mut byte);
    unsafe { *out = event };
    1
}

fn input_config(index: usize) -> InputConfig {
    InputConfig {
        features: 1, // KRUN_INPUT_CONFIG_FEATURE_QUERY
        create_userdata: index as *const c_void,
        create,
        destroy: std::ptr::null(),
        query_device_name: device_name,
        query_serial_name: serial_name,
        query_device_ids: device_ids,
        query_event_capabilities: event_capabilities,
        query_abs_info: abs_info,
        query_properties: properties,
    }
}

fn input_events(index: usize) -> InputEventProvider {
    InputEventProvider {
        features: 1, // KRUN_INPUT_EVENT_PROVIDER_FEATURE_QUEUE
        create_userdata: index as *const c_void,
        create,
        destroy: std::ptr::null(),
        get_ready_efd: ready_fd,
        next_event,
    }
}

/// Reads input records from the socket into the device queues.
fn read_input(mut socket: UnixStream) {
    let mut record = [0u8; 12];
    while socket.read_exact(&mut record).is_ok() {
        let d = u16::from_le_bytes([record[0], record[1]]) as usize;
        if d == 2 {
            let value = u32::from_le_bytes([record[8], record[9], record[10], record[11]]);
            *channel().pending_size.lock().unwrap() = Some((value >> 16, value & 0xffff));
            let _ = (&channel().resize_pipe.1).write_all(&[0]);
            continue;
        }
        if d > 1 {
            continue;
        }
        let event = InputEvent {
            type_: u16::from_le_bytes([record[2], record[3]]),
            code: u16::from_le_bytes([record[4], record[5]]),
            value: u32::from_le_bytes([record[8], record[9], record[10], record[11]]),
        };
        channel().queues[d].lock().unwrap().push_back(event);
        let _ = (&channel().pipes[d].1).write_all(&[0]);
    }
}

pub fn setup(
    ctx_id: u32,
    path: &Path,
    width: u32,
    height: u32,
    dpi: Option<u32>,
    refresh_rate: Option<u32>,
) -> Result<()> {
    let socket = UnixStream::connect(path).with_context(|| format!("connecting to {}", path.display()))?;
    let reader = socket.try_clone()?;
    let pipe = || -> Result<(File, File)> {
        let (r, w) = std::io::pipe()?;
        Ok((File::from(std::os::fd::OwnedFd::from(r)), File::from(std::os::fd::OwnedFd::from(w))))
    };
    let mut frames_path = path.as_os_str().to_owned();
    frames_path.push(".frames");
    let channel = Channel {
        socket: Mutex::new(socket),
        frames: Mutex::new(None),
        queues: [Mutex::new(VecDeque::new()), Mutex::new(VecDeque::new())],
        pipes: [pipe()?, pipe()?],
        pending_size: Mutex::new(None),
        resize_pipe: pipe()?,
        blobs: Mutex::new(HashSet::new()),
    };
    // configure_scanout reads the path from the empty slot.
    *channel.frames.lock().unwrap() = Some(Frames {
        path: PathBuf::from(frames_path),
        map: std::ptr::null_mut(),
        buffer_size: 0,
        next: 0,
        shape: (0, 0, 0),
    });
    CHANNEL.set(channel).map_err(|_| anyhow!("display socket set up twice"))?;
    std::thread::spawn(move || read_input(reader));

    unsafe {
        if krun_add_display(ctx_id, width, height) < 0 {
            return Err(anyhow!("krun_add_display failed"));
        }
        if krun_set_display_backend(ctx_id, &DISPLAY as *const _ as *const c_void, std::mem::size_of::<DisplayBackend>()) < 0 {
            return Err(anyhow!("krun_set_display_backend failed"));
        }
        if let Some(dpi) = dpi {
            if krun_display_set_dpi(ctx_id, 0, dpi) < 0 {
                return Err(anyhow!("krun_display_set_dpi failed"));
            }
        }
        if let Some(rate) = refresh_rate {
            if krun_display_set_refresh_rate(ctx_id, 0, rate) < 0 {
                return Err(anyhow!("krun_display_set_refresh_rate failed"));
            }
        }
        for index in 0..2 {
            let config = Box::leak(Box::new(input_config(index)));
            let events = Box::leak(Box::new(input_events(index)));
            if krun_add_input_device(
                ctx_id,
                config as *const _ as *const c_void,
                std::mem::size_of::<InputConfig>(),
                events as *const _ as *const c_void,
                std::mem::size_of::<InputEventProvider>(),
            ) < 0
            {
                return Err(anyhow!("krun_add_input_device failed"));
            }
        }
    }
    Ok(())
}
