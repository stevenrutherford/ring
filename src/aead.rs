// Copyright 2015-2016 Brian Smith.
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHORS DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
// SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
// OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
// CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

//! Authenticated Encryption with Associated Data (AEAD).
//!
//! See [Authenticated encryption: relations among notions and analysis of the
//! generic composition paradigm][AEAD] for an introduction to the concept of
//! AEADs.
//!
//! [AEAD]: http://www-cse.ucsd.edu/~mihir/papers/oem.html
//! [`crypto.cipher.AEAD`]: https://golang.org/pkg/crypto/cipher/#AEAD

use self::block::{Block, BLOCK_LEN};
use crate::{constant_time, cpu, error, hkdf, polyfill};
use core::ops::{RangeFrom, RangeTo};

pub use self::{
    aes_gcm::{AES_128_GCM, AES_256_GCM},
    chacha20_poly1305::CHACHA20_POLY1305,
    nonce::{Nonce, NONCE_LEN},
};

/// A sequences of unique nonces.
///
/// A given `NonceSequence` must never return the same `Nonce` twice from
/// `advance()`.
///
/// A simple counter is a reasonable (but probably not ideal) `NonceSequence`.
///
/// Intentionally not `Clone` or `Copy` since cloning would allow duplication
/// of the sequence.
pub trait NonceSequence {
    /// Returns the next nonce in the sequence.
    ///
    /// This may fail if "too many" nonces have been requested, where how many
    /// is too many is up to the implementation of `NonceSequence`. An
    /// implementation may that enforce a maximum number of records are
    /// sent/received under a key this way. Once `advance()` fails, it must
    /// fail for all subsequent calls.
    fn advance(&mut self) -> Result<Nonce, error::Unspecified>;
}

/// An AEAD key bound to a nonce sequence.
pub trait BoundKey<N: NonceSequence>: core::fmt::Debug {
    /// Constructs a new key from the given `UnboundKey` and `NonceSequence`.
    fn new(key: UnboundKey, nonce_sequence: N) -> Self;

    /// The key's AEAD algorithm.
    fn algorithm(&self) -> &'static Algorithm;
}

/// An AEAD key for authenticating and decrypting ("opening"), bound to a nonce
/// sequence.
///
/// Intentionally not `Clone` or `Copy` since cloning would allow duplication
/// of the nonce sequence.
pub struct OpeningKey<N: NonceSequence> {
    key: UnboundKey,
    nonce_sequence: N,
}

impl<N: NonceSequence> BoundKey<N> for OpeningKey<N> {
    fn new(key: UnboundKey, nonce_sequence: N) -> Self {
        Self {
            key,
            nonce_sequence,
        }
    }

    #[inline]
    fn algorithm(&self) -> &'static Algorithm {
        self.key.algorithm
    }
}

impl<N: NonceSequence> core::fmt::Debug for OpeningKey<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("OpeningKey")
            .field("algorithm", &self.algorithm())
            .finish()
    }
}

impl<N: NonceSequence> OpeningKey<N> {
    /// Authenticates and decrypts (“opens”) data in place.
    ///
    /// `aad` is the additional authenticated data (AAD), if any.
    ///
    /// On input, `in_out` must be the ciphertext followed by the tag. When
    /// `open_in_place()` returns `Ok(plaintext)`, the input ciphertext
    /// has been overwritten by the plaintext; `plaintext` will refer to the
    /// plaintext without the tag.
    ///
    /// When `open_in_place()` returns `Err(..)`, `in_out` may have been
    /// overwritten in an unspecified way.
    #[inline]
    pub fn open_in_place<'in_out, A>(
        &mut self,
        aad: Aad<A>,
        in_out: &'in_out mut [u8],
    ) -> Result<&'in_out mut [u8], error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        self.open_within(aad, in_out, 0..)
    }

    /// Authenticates and decrypts (“opens”) data in place, with a shift.
    ///
    /// `aad` is the additional authenticated data (AAD), if any.
    ///
    /// On input, `in_out[ciphertext_and_tag]` must be the ciphertext followed
    /// by the tag. When `open_within()` returns `Ok(plaintext)`, the plaintext
    /// will be at `in_out[0..plaintext.len()]`. In other words, the following
    /// two code fragments are equivalent for valid values of
    /// `ciphertext_and_tag`, except `open_within` will often be more efficient:
    ///
    ///
    /// ```skip
    /// let plaintext = key.open_within(aad, in_out, cipertext_and_tag)?;
    /// ```
    ///
    /// ```skip
    /// let ciphertext_and_tag_len = in_out[ciphertext_and_tag].len();
    /// in_out.copy_within(ciphertext_and_tag, 0);
    /// let plaintext = key.open_in_place(aad, &mut in_out[..ciphertext_and_tag_len])?;
    /// ```
    ///
    /// Similarly, `key.open_within(aad, in_out, 0..)` is equivalent to
    /// `key.open_in_place(aad, in_out)`.
    ///
    ///  When `open_in_place()` returns `Err(..)`, `in_out` may have been
    /// overwritten in an unspecified way.
    ///
    /// The shifting feature is useful in the case where multiple packets are
    /// being reassembled in place. Consider this example where the peer has
    /// sent the message “Split stream reassembled in place” split into
    /// three sealed packets:
    ///
    /// ```ascii-art
    ///                 Packet 1                  Packet 2                 Packet 3
    /// Input:  [Header][Ciphertext][Tag][Header][Ciphertext][Tag][Header][Ciphertext][Tag]
    ///                      |         +--------------+                        |
    ///               +------+   +-----+    +----------------------------------+
    ///               v          v          v
    /// Output: [Plaintext][Plaintext][Plaintext]
    ///        “Split stream reassembled in place”
    /// ```
    ///
    /// This reassembly be accomplished with three calls to `open_within()`.
    #[inline]
    pub fn open_within<'in_out, A>(
        &mut self,
        aad: Aad<A>,
        in_out: &'in_out mut [u8],
        ciphertext_and_tag: RangeFrom<usize>,
    ) -> Result<&'in_out mut [u8], error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        open_within_(
            &self.key,
            self.nonce_sequence.advance()?,
            aad,
            in_out,
            ciphertext_and_tag,
        )
    }
}

fn open_<'in_out>(
    key: &UnboundKey,
    nonce: Nonce,
    aad: Aad<&[u8]>,
    mut in_out: IOBuffer<'in_out>,
    received_tag: &[u8],
) -> Result<(), error::Unspecified> {
    check_per_nonce_max_bytes(key.algorithm, in_out.len())?;
    let Tag(calculated_tag) =
        (key.algorithm.open)(&key.inner, nonce, aad, in_out.borrow(), key.cpu_features);
    if constant_time::verify_slices_are_equal(calculated_tag.as_ref(), received_tag).is_err() {
        // Zero out the plaintext so that it isn't accidentally leaked or used
        // after verification fails. It would be safest if we could check the
        // tag before decrypting, but some `open` implementations interleave
        // authentication with decryption for performance.
        let mut in_out = in_out.borrow();
        for b in &mut in_out.output().iter_mut() {
            *b = 0;
        }
        return Err(error::Unspecified);
    }
    Ok(())
}

fn open_within_<'in_out, A: AsRef<[u8]>>(
    key: &UnboundKey,
    nonce: Nonce,
    Aad(aad): Aad<A>,
    in_out: &'in_out mut [u8],
    ciphertext_and_tag: RangeFrom<usize>,
) -> Result<&'in_out mut [u8], error::Unspecified> {
    let in_prefix_len = ciphertext_and_tag.start;
    let ciphertext_and_tag_len = in_out
        .len()
        .checked_sub(in_prefix_len)
        .ok_or(error::Unspecified)?;
    let ciphertext_len = ciphertext_and_tag_len
        .checked_sub(TAG_LEN)
        .ok_or(error::Unspecified)?;
    let (in_out, received_tag) = in_out.split_at_mut(in_prefix_len + ciphertext_len);
    let mut iobuffer = IOBuffer::new_in_place_offset(in_out, in_prefix_len)?;
    open_(
        key,
        nonce,
        Aad::from(aad.as_ref()),
        iobuffer.borrow(),
        received_tag,
    )?;
    Ok(&mut in_out[..ciphertext_len])
}

/// An AEAD key for encrypting and signing ("sealing"), bound to a nonce
/// sequence.
///
/// Intentionally not `Clone` or `Copy` since cloning would allow duplication
/// of the nonce sequence.
pub struct SealingKey<N: NonceSequence> {
    key: UnboundKey,
    nonce_sequence: N,
}

impl<N: NonceSequence> BoundKey<N> for SealingKey<N> {
    fn new(key: UnboundKey, nonce_sequence: N) -> Self {
        Self {
            key,
            nonce_sequence,
        }
    }

    #[inline]
    fn algorithm(&self) -> &'static Algorithm {
        self.key.algorithm
    }
}

impl<N: NonceSequence> core::fmt::Debug for SealingKey<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("SealingKey")
            .field("algorithm", &self.algorithm())
            .finish()
    }
}

impl<N: NonceSequence> SealingKey<N> {
    /// Deprecated. Renamed to `seal_in_place_append_tag()`.
    #[deprecated(note = "Renamed to `seal_in_place_append_tag`.")]
    #[inline]
    pub fn seal_in_place<A, InOut>(
        &mut self,
        aad: Aad<A>,
        in_out: &mut InOut,
    ) -> Result<(), error::Unspecified>
    where
        A: AsRef<[u8]>,
        InOut: AsMut<[u8]> + for<'in_out> Extend<&'in_out u8>,
    {
        self.seal_in_place_append_tag(aad, in_out)
    }

    /// Encrypts and signs (“seals”) data in place, appending the tag to the
    /// resulting ciphertext.
    ///
    /// `key.seal_in_place_append_tag(aad, in_out)` is equivalent to:
    ///
    /// ```skip
    /// key.seal_in_place_separate_tag(aad, in_out.as_mut())
    ///     .map(|tag| in_out.extend(tag.as_ref()))
    /// ```
    #[inline]
    pub fn seal_in_place_append_tag<A, InOut>(
        &mut self,
        aad: Aad<A>,
        in_out: &mut InOut,
    ) -> Result<(), error::Unspecified>
    where
        A: AsRef<[u8]>,
        InOut: AsMut<[u8]> + for<'in_out> Extend<&'in_out u8>,
    {
        self.seal_in_place_separate_tag(aad, in_out.as_mut())
            .map(|tag| in_out.extend(tag.as_ref()))
    }

    /// Encrypts and signs (“seals”) data in place.
    ///
    /// `aad` is the additional authenticated data (AAD), if any. This is
    /// authenticated but not encrypted. The type `A` could be a byte slice
    /// `&[u8]`, a byte array `[u8; N]` for some constant `N`, `Vec<u8>`, etc.
    /// If there is no AAD then use `Aad::empty()`.
    ///
    /// The plaintext is given as the input value of `in_out`. `seal_in_place()`
    /// will overwrite the plaintext with the ciphertext and return the tag.
    /// For most protocols, the caller must append the tag to the ciphertext.
    /// The tag will be `self.algorithm.tag_len()` bytes long.
    #[inline]
    pub fn seal_in_place_separate_tag<A>(
        &mut self,
        aad: Aad<A>,
        in_out: &mut [u8],
    ) -> Result<Tag, error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        seal_in_place_separate_tag_(
            &self.key,
            self.nonce_sequence.advance()?,
            Aad::from(aad.as_ref()),
            in_out,
        )
    }
}

#[inline]
fn seal_in_place_separate_tag_(
    key: &UnboundKey,
    nonce: Nonce,
    aad: Aad<&[u8]>,
    in_out: &mut [u8],
) -> Result<Tag, error::Unspecified> {
    check_per_nonce_max_bytes(key.algorithm, in_out.len())?;
    let iobuffer = IOBuffer::new_in_place(in_out);
    Ok((key.algorithm.seal)(
        &key.inner,
        nonce,
        aad,
        iobuffer,
        key.cpu_features,
    ))
}

/// The additionally authenticated data (AAD) for an opening or sealing
/// operation. This data is authenticated but is **not** encrypted.
///
/// The type `A` could be a byte slice `&[u8]`, a byte array `[u8; N]`
/// for some constant `N`, `Vec<u8>`, etc.
pub struct Aad<A: AsRef<[u8]>>(A);

impl<A: AsRef<[u8]>> Aad<A> {
    /// Construct the `Aad` from the given bytes.
    #[inline]
    pub fn from(aad: A) -> Self {
        Aad(aad)
    }
}

impl<A> AsRef<[u8]> for Aad<A>
where
    A: AsRef<[u8]>,
{
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl Aad<[u8; 0]> {
    /// Construct an empty `Aad`.
    pub fn empty() -> Self {
        Self::from([])
    }
}

/// An AEAD key without a designated role or nonce sequence.
pub struct UnboundKey {
    inner: KeyInner,
    algorithm: &'static Algorithm,
    cpu_features: cpu::Features,
}

impl core::fmt::Debug for UnboundKey {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("UnboundKey")
            .field("algorithm", &self.algorithm)
            .finish()
    }
}

#[allow(variant_size_differences)]
enum KeyInner {
    AesGcm(aes_gcm::Key),
    ChaCha20Poly1305(chacha20_poly1305::Key),
}

impl UnboundKey {
    /// Constructs an `UnboundKey`.
    ///
    /// Fails if `key_bytes.len() != algorithm.key_len()`.
    pub fn new(
        algorithm: &'static Algorithm,
        key_bytes: &[u8],
    ) -> Result<Self, error::Unspecified> {
        let cpu_features = cpu::features();
        Ok(Self {
            inner: (algorithm.init)(key_bytes, cpu_features)?,
            algorithm,
            cpu_features,
        })
    }

    /// The key's AEAD algorithm.
    #[inline]
    pub fn algorithm(&self) -> &'static Algorithm {
        self.algorithm
    }
}

impl From<hkdf::Okm<'_, &'static Algorithm>> for UnboundKey {
    fn from(okm: hkdf::Okm<&'static Algorithm>) -> Self {
        let mut key_bytes = [0; MAX_KEY_LEN];
        let key_bytes = &mut key_bytes[..okm.len().key_len];
        let algorithm = *okm.len();
        okm.fill(key_bytes).unwrap();
        Self::new(algorithm, key_bytes).unwrap()
    }
}

impl hkdf::KeyType for &'static Algorithm {
    #[inline]
    fn len(&self) -> usize {
        self.key_len()
    }
}

/// Immutable keys for use in situations where `OpeningKey`/`SealingKey` and
/// `NonceSequence` cannot reasonably be used.
///
/// Prefer to use `OpeningKey`/`SealingKey` and `NonceSequence` when practical.
pub struct LessSafeKey {
    key: UnboundKey,
}

impl LessSafeKey {
    /// Constructs a `LessSafeKey` from an `UnboundKey`.
    pub fn new(key: UnboundKey) -> Self {
        Self { key }
    }

    /// Like [`OpeningKey::open_in_place()`], except it accepts an arbitrary nonce.
    ///
    /// `nonce` must be unique for every use of the key to open data.
    #[inline]
    pub fn open_in_place<'in_out, A>(
        &self,
        nonce: Nonce,
        aad: Aad<A>,
        in_out: &'in_out mut [u8],
    ) -> Result<&'in_out mut [u8], error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        self.open_within(nonce, aad, in_out, 0..)
    }

    /// Like [`OpeningKey::open_within()`], except it accepts an arbitrary nonce.
    ///
    /// `nonce` must be unique for every use of the key to open data.
    #[inline]
    pub fn open_within<'in_out, A>(
        &self,
        nonce: Nonce,
        aad: Aad<A>,
        in_out: &'in_out mut [u8],
        ciphertext_and_tag: RangeFrom<usize>,
    ) -> Result<&'in_out mut [u8], error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        open_within_(&self.key, nonce, aad, in_out, ciphertext_and_tag)
    }

    /// Deprecated. Renamed to `seal_in_place_append_tag()`.
    #[deprecated(note = "Renamed to `seal_in_place_append_tag`.")]
    #[inline]
    pub fn seal_in_place<A, InOut>(
        &self,
        nonce: Nonce,
        aad: Aad<A>,
        in_out: &mut InOut,
    ) -> Result<(), error::Unspecified>
    where
        A: AsRef<[u8]>,
        InOut: AsMut<[u8]> + for<'in_out> Extend<&'in_out u8>,
    {
        self.seal_in_place_append_tag(nonce, aad, in_out)
    }

    /// Like [`SealingKey::seal_in_place_append_tag()`], except it accepts an
    /// arbitrary nonce.
    ///
    /// `nonce` must be unique for every use of the key to seal data.
    #[inline]
    pub fn seal_in_place_append_tag<A, InOut>(
        &self,
        nonce: Nonce,
        aad: Aad<A>,
        in_out: &mut InOut,
    ) -> Result<(), error::Unspecified>
    where
        A: AsRef<[u8]>,
        InOut: AsMut<[u8]> + for<'in_out> Extend<&'in_out u8>,
    {
        self.seal_in_place_separate_tag(nonce, aad, in_out.as_mut())
            .map(|tag| in_out.extend(tag.as_ref()))
    }

    /// Like `SealingKey::seal_in_place_separate_tag()`, except it accepts an
    /// arbitrary nonce.
    ///
    /// `nonce` must be unique for every use of the key to seal data.
    #[inline]
    pub fn seal_in_place_separate_tag<A>(
        &self,
        nonce: Nonce,
        aad: Aad<A>,
        in_out: &mut [u8],
    ) -> Result<Tag, error::Unspecified>
    where
        A: AsRef<[u8]>,
    {
        seal_in_place_separate_tag_(&self.key, nonce, Aad::from(aad.as_ref()), in_out)
    }

    /// The key's AEAD algorithm.
    #[inline]
    pub fn algorithm(&self) -> &'static Algorithm {
        &self.key.algorithm
    }
}

impl core::fmt::Debug for LessSafeKey {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("LessSafeKey")
            .field("algorithm", self.algorithm())
            .finish()
    }
}

/// An AEAD Algorithm.
pub struct Algorithm {
    init: fn(key: &[u8], cpu_features: cpu::Features) -> Result<KeyInner, error::Unspecified>,

    seal: fn(
        key: &KeyInner,
        nonce: Nonce,
        aad: Aad<&[u8]>,
        in_out: IOBuffer<'_>,
        cpu_features: cpu::Features,
    ) -> Tag,
    open: fn(
        key: &KeyInner,
        nonce: Nonce,
        aad: Aad<&[u8]>,
        in_out: IOBuffer<'_>,
        cpu_features: cpu::Features,
    ) -> Tag,

    key_len: usize,
    id: AlgorithmID,

    /// Use `max_input_len!()` to initialize this.
    // TODO: Make this `usize`.
    max_input_len: u64,
}

const fn max_input_len(block_len: usize, overhead_blocks_per_nonce: usize) -> u64 {
    // Each of our AEADs use a 32-bit block counter so the maximum is the
    // largest input that will not overflow the counter.
    ((1u64 << 32) - polyfill::u64_from_usize(overhead_blocks_per_nonce))
        * polyfill::u64_from_usize(block_len)
}

impl Algorithm {
    /// The length of the key.
    #[inline(always)]
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    /// The length of a tag.
    ///
    /// See also `MAX_TAG_LEN`.
    #[inline(always)]
    pub fn tag_len(&self) -> usize {
        TAG_LEN
    }

    /// The length of the nonces.
    #[inline(always)]
    pub fn nonce_len(&self) -> usize {
        NONCE_LEN
    }
}

derive_debug_via_id!(Algorithm);

#[derive(Debug, Eq, PartialEq)]
enum AlgorithmID {
    AES_128_GCM,
    AES_256_GCM,
    CHACHA20_POLY1305,
}

impl PartialEq for Algorithm {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Algorithm {}

/// An authentication tag.
#[must_use]
#[repr(C)]
pub struct Tag([u8; TAG_LEN]);

impl AsRef<[u8]> for Tag {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

const MAX_KEY_LEN: usize = 32;

// All the AEADs we support use 128-bit tags.
const TAG_LEN: usize = BLOCK_LEN;

/// The maximum length of a tag for the algorithms in this module.
pub const MAX_TAG_LEN: usize = TAG_LEN;

fn check_per_nonce_max_bytes(alg: &Algorithm, in_out_len: usize) -> Result<(), error::Unspecified> {
    if polyfill::u64_from_usize(in_out_len) > alg.max_input_len {
        return Err(error::Unspecified);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Direction {
    Opening,
    Sealing,
}

// TODO: ideally this would not need to be public, but chacha's encrypt, and
// shift_partial leak this type.
/// IOBuffer is an abstraction for Input/Output buffers used in AEAD.
#[derive(Debug)]
pub enum IOBuffer<'in_out> {
    /// InPlace is the wrapper for a mutable in_out buffer and an in_prefix_len.
    InPlace {
        /// In/Out Buffer for AEAD.
        in_out: &'in_out mut [u8],
        /// Length of prefix before the ciphertext. See `open_within` for more.
        in_prefix_len: usize,
    },
    /// Separate is the wrapper for an input buffer and a mutable output buffer.
    Separate {
        /// Input buffer for AEAD.
        input: &'in_out [u8],
        /// Output buffer for AEAD.
        output: &'in_out mut [u8],
    },
}

impl<'in_out> IOBuffer<'in_out> {
    fn new_in_place(in_out: &'in_out mut [u8]) -> IOBuffer {
        IOBuffer::InPlace {
            in_out,
            in_prefix_len: 0,
        }
    }
    fn new_in_place_offset(
        in_out: &'in_out mut [u8],
        in_prefix_len: usize,
    ) -> Result<IOBuffer, error::Unspecified> {
        if in_out.len() < in_prefix_len {
            return Err(error::Unspecified);
        }
        Ok(IOBuffer::InPlace {
            in_out,
            in_prefix_len,
        })
    }
    fn _new_separate(
        input: &'in_out [u8],
        output: &'in_out mut [u8],
    ) -> Result<IOBuffer<'in_out>, error::Unspecified> {
        if input.len() != output.len() {
            return Err(error::Unspecified);
        }
        Ok(IOBuffer::Separate { input, output })
    }

    #[inline]
    fn input(&self) -> &[u8] {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len,
            } => &in_out[*in_prefix_len..],
            IOBuffer::Separate { input, output: _ } => input,
        }
    }
    #[inline]
    fn output(&mut self) -> &mut [u8] {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len: _,
            } => *in_out,
            IOBuffer::Separate { input: _, output } => *output,
        }
    }
    #[inline]
    fn len(&self) -> usize {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len,
            } => in_out.len() - in_prefix_len,
            IOBuffer::Separate { input: _, output } => output.len(),
        }
    }
    #[inline]
    fn borrow<'buf>(&'buf mut self) -> IOBuffer<'buf> {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len,
            } => IOBuffer::InPlace {
                in_out,
                in_prefix_len: *in_prefix_len,
            },
            IOBuffer::Separate { input, output } => IOBuffer::Separate { input, output },
        }
    }
    #[inline]
    fn index_to(self, idx: RangeTo<usize>) -> IOBuffer<'in_out> {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len,
            } => IOBuffer::InPlace {
                // Need to ensure that the in_out buffer spans the full width
                // from start of input to end of output.
                in_out: &mut in_out[..idx.end + in_prefix_len],
                in_prefix_len,
            },
            IOBuffer::Separate { input, output } => IOBuffer::Separate {
                input: &input[idx.clone()],
                output: &mut output[idx],
            },
        }
    }
    #[inline]
    fn index_from(self, idx: RangeFrom<usize>) -> IOBuffer<'in_out> {
        match self {
            IOBuffer::InPlace {
                in_out,
                in_prefix_len,
            } => IOBuffer::InPlace {
                in_out: &mut in_out[idx],
                in_prefix_len,
            },
            IOBuffer::Separate { input, output } => IOBuffer::Separate {
                input: &input[idx.clone()],
                output: &mut output[idx],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_separate_buffer() {
        let input = [0, 1, 2];
        let mut output = [0, 1, 2];
        let mut split = IOBuffer::_new_separate(&input, &mut output).unwrap();
        assert_eq!(split.borrow().index_to(..2).input(), [0, 1]);
        assert_eq!(split.borrow().index_to(..2).output(), [0, 1]);
        assert_eq!(split.borrow().index_from(1..).input(), [1, 2]);
        assert_eq!(split.borrow().index_from(1..).output(), [1, 2]);
        assert_eq!(split.borrow().index_to(..2).index_from(1..).input(), [1]);
        assert_eq!(split.borrow().index_to(..2).index_from(1..).output(), [1]);
    }
}

mod aes;
mod aes_gcm;
mod block;
mod chacha;
mod chacha20_poly1305;
pub mod chacha20_poly1305_openssh;
mod counter;
mod gcm;
mod iv;
mod nonce;
mod poly1305;
pub mod quic;
mod shift;
